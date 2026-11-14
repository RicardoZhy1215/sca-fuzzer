#!/bin/bash
# Transfer test: a ctxrl (GRU) Spectre v4 policy trained on one machine, used on another.
# On the TARGET machine, same config (v4.yaml knobs, 64 instructions, SSBD off, -i 50) and the same
# number of programs per arm, one arm after the other:
#   revizor  unmodified Revizor
#   cold     ctxrl from scratch (Revizor's prior), learning on this machine (= run_compare.sh v4)
#   warm     ctxrl starting from the transferred policy, still learning
#   frozen   the transferred policy, never updated (--rl-lr 0): zero-shot transfer
# The policy (default transfer/v4-i7-6700/policy.pt, 20000 programs on an i7-6700) is loaded only
# if this machine builds the same instruction table (ctxrl.py checks it against the table.json next
# to the policy), so a different base.json / config / generator aborts instead of mis-mapping.
#
#   cd <sca-fuzzer checkout>
#   PY=<python with Revizor deps + torch> [PLOTPY=...] [ARMS="revizor cold warm frozen"] [POLICY=...] \
#       rvzr/SpecRL/genbandit/transfer/run_transfer.sh <N> [fuzzer cores (1-3)] [learner cores (4-7)]
# Output: transfer/runs/v4-<host>-<cpu>-<N>-<date>/{revizor,cold,warm,frozen}/, launch.log,
#         machine.txt, RESULTS.txt (transfer/summarize.py); figure transfer/figs/<same name>.*
# Runs detached; waits on genbandit/.executor.lock and for any other fuzzer (never two at once).
set -eu
N=${1:?programs per arm}; FC=${2:-1-3}; LC=${3:-4-7}
ROOT=$(git rev-parse --show-toplevel)
G=$ROOT/rvzr/SpecRL/genbandit
T=$G/transfer
PY=$(readlink -f "${PY:-$(command -v python)}")
PLOTPY=$(readlink -f "${PLOTPY:-$PY}")
ARMS=${ARMS:-revizor cold warm frozen}
POLICY=$(readlink -f "${POLICY:-$T/v4-i7-6700/policy.pt}")
[ -f "$POLICY" ] || { echo "no policy: $POLICY"; exit 1; }
ACTION=$("$PY" -c "import sys, json; sys.path.insert(0, '$G'); import space_v4 as s; print(json.dumps({**s.CONTROL, 'program_size': 64}))")
CFG=$("$PY" -c "import sys, json; sys.path.insert(0, '$G'); import space_v4 as s; print(s.write(json.loads('$ACTION')))")
if [ -f "$(dirname $POLICY)/config.yaml" ] && ! cmp -s "$CFG" "$(dirname $POLICY)/config.yaml"; then
    echo "config $CFG differs from the one the policy was trained with ($(dirname $POLICY)/config.yaml)"; exit 1
fi
SRC=$(grep -m1 "model name" "$(dirname $POLICY)/machine.txt" 2>/dev/null | sed 's/.*: //; s/(R)\|(TM)\| CPU @.*//g' || true)
TGT=$(grep -m1 "model name" /proc/cpuinfo | sed 's/.*: //; s/(R)\|(TM)\| CPU @.*//g')
CPU=$(grep -m1 "model name" /proc/cpuinfo | sed 's/.*: //; s/[^A-Za-z0-9]\+/_/g' | cut -c1-40)
NAME=v4-$(hostname -s)-$CPU-$N-$(date +%y%m%d-%H%M)
R=$T/runs/$NAME
mkdir -p $R $T/figs
for a in $ARMS; do mkdir -p $R/$a; done
cp "$CFG" $R/config.yaml
SERIES=""
for a in $ARMS; do
    case $a in revizor) L=Revizor ;; cold) L="RL from scratch" ;; warm) L="RL transferred (learning)" ;;
               frozen) L="RL transferred (frozen)" ;; *) echo "unknown arm $a"; exit 1 ;; esac
    SERIES="$SERIES --series \"$L=$R/$a\""
done
TCENV="$G/tcenv.py --space v4 --action '$ACTION' --mode bypass --minutes 4320 --max-programs $N \
--policy ctx --ctx-policy learn --ctx-mode rule2 --learner-cores $LC"
cat > $R/launch.sh <<EOF
#!/bin/bash
cd $ROOT
export PYTHONPATH=$ROOT
LOG=$R/launch.log
echo "queued \$(date)" > \$LOG
exec 9>$G/.executor.lock
flock 9
while pgrep -f "^[^ ]*python[^ ]* ([^ ]*/)?(revizor\.py fuzz|tcenv\.py|ttk_bench\.py)" > /dev/null; do sleep 30; done
if [ "\$(cat /proc/sys/kernel/nmi_watchdog)" != "0" ]; then echo "nmi_watchdog is on: abort" >> \$LOG; exit 1; fi
{ echo "host \$(hostname)"; grep -m1 "model name" /proc/cpuinfo; grep -m1 microcode /proc/cpuinfo; uname -r
  echo "smt \$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)"
  echo "spec_store_bypass: \$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)"
  cat /proc/cmdline; git rev-parse HEAD
  echo "fuzzer cores $FC, learner cores $LC, python $PY"; echo "policy $POLICY \$(md5sum < $POLICY | cut -c1-32)"
  echo "arms $ARMS, $N programs each"; } > $R/machine.txt
for a in $ARMS; do
    echo "\$a start \$(date +%s) \$(date)" >> \$LOG
    case \$a in
      revizor) taskset -c $FC $PY revizor.py fuzz -s base.json -c $R/config.yaml -I $R/revizor -n $N -i 50 \\
                   -w $R/revizor --nonstop > $R/revizor/fuzz.log 2>&1 ;;
      cold)    taskset -c $FC $PY $TCENV --out $R/cold > $R/cold/run.log 2>&1 ;;
      warm)    taskset -c $FC $PY $TCENV --out $R/warm --ctx-init $POLICY > $R/warm/run.log 2>&1 ;;
      frozen)  taskset -c $FC $PY $TCENV --out $R/frozen --ctx-init $POLICY --rl-lr 0 > $R/frozen/run.log 2>&1 ;;
    esac
    echo "\$a end \$(date) rc \$?" >> \$LOG
done
$PLOTPY $G/spectre_v1/plot_curve.py --out $T/figs/$NAME --no-grid --no-joins \\
    --title "Spectre v4 transfer: cumulative violations vs programs" \\
    --subtitle "policy trained on ${SRC:-another machine} · run on $TGT" $SERIES >> \$LOG 2>&1
$PY $T/summarize.py $R > $R/RESULTS.txt 2>> \$LOG
echo "done \$(date)" >> \$LOG
EOF
# the plot script counts Revizor's last -n program from a launch.sh in the run dir
[ -d $R/revizor ] && printf '#!/bin/bash\n# (see ../launch.sh) revizor.py fuzz ... -n %s -i 50\n' "$N" > $R/revizor/launch.sh
chmod +x $R/launch.sh
(cd $R && setsid nohup ./launch.sh > launch.out 2>&1 < /dev/null &)
echo "started: $R"
echo "progress: tail $R/launch.log; result: $R/RESULTS.txt and $T/figs/$NAME.png"
