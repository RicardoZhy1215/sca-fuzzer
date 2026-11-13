#!/bin/bash
# Tier A: unmodified Revizor, then ctxrl (GRU, learned modes), N programs each, on the SAME config
# (tier_a/configs/<name>.yaml or a path); then the cumulative-violations figure. CPU-agnostic (Revizor
# picks the Intel or AMD executor from /proc/cpuinfo). Runs detached unless INLINE=1.
#
#   cd <sca-fuzzer checkout>
#   PY=<python with Revizor's deps + torch> rvzr/SpecRL/genbandit/tier_a/run_compare.sh <config> <N> [fuzzer cores] [learner cores]
#   e.g.  rvzr/SpecRL/genbandit/tier_a/run_compare.sh de-dh 10000              # cores 1-3 / 4-7
#         rvzr/SpecRL/genbandit/tier_a/run_compare.sh zdi 20000 0-1 2-5        # fewer cores
#   ARMS=revizor ... runs only Revizor (e.g. a novelty check); ARMS=rl only ctxrl.
# On a hybrid Intel CPU (P- and E-cores) give P-cores only: E-cores are a different microarchitecture.
# Output: tier_a/runs/<config>-<host>-<cpu>-<N>-<date>/{revizor,ctx}/, figure tier_a/figs/<same name>.*
set -eu
C=${1:?config name (tier_a/configs/<name>.yaml) or path}; N=${2:?programs per tool}; FC=${3:-1-3}; LC=${4:-4-7}
ARMS=${ARMS:-revizor,rl}
ROOT=$(git rev-parse --show-toplevel)
T=$ROOT/rvzr/SpecRL/genbandit/tier_a
[ -f "$C" ] || C=$T/configs/$C.yaml
C=$(readlink -f "$C")
BASE=$(basename "$C" .yaml)
PY=$(readlink -f "${PY:-$(command -v python)}")
CPU=$(grep -m1 "model name" /proc/cpuinfo | sed 's/.*: //; s/[^A-Za-z0-9]\+/_/g' | cut -c1-40)
NAME=$BASE-$(hostname -s)-$CPU-$N-$(date +%y%m%d-%H%M)
R=$T/runs/$NAME
mkdir -p $R/revizor $R/ctx $T/figs
CFG=$("$PY" -c "import sys; sys.path.insert(0, '$T/..'); sys.path.insert(0, '$T'); import space_yaml as s; s.configure('$C'); print(s.write(s.CONTROL))")
cp "$CFG" $R/config.yaml
cp "$CFG" $R/revizor/config.yaml
cat > $R/launch.sh <<EOF
#!/bin/bash
cd $ROOT
LOG=$R/launch.log
{ echo "host \$(hostname)"; grep -m1 "model name" /proc/cpuinfo; grep -m1 microcode /proc/cpuinfo; uname -r
  echo "smt \$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)"; cat /proc/cmdline; git rev-parse HEAD; } > $R/machine.txt
exec 9>$T/.executor.lock
flock 9
while pgrep -f "^[^ ]*python[^ ]* ([^ ]*/)?(revizor\.py fuzz|tcenv\.py|ttk_bench\.py)" > /dev/null; do sleep 30; done
if [[ "$ARMS" == *revizor* ]]; then
  echo "revizor start \$(date)" >> \$LOG
  taskset -c $FC $PY revizor.py fuzz -s base.json -c $R/revizor/config.yaml -n $N -i 50 -w $R/revizor --nonstop > $R/revizor/fuzz.log 2>&1 || true
  echo "revizor end \$(date)" >> \$LOG
fi
if [[ "$ARMS" == *rl* ]]; then
  echo "rl start \$(date)" >> \$LOG
  taskset -c $FC $PY $T/../tcenv.py --space yaml --space-yaml $R/config.yaml --mode bypass --minutes 4320 \\
      --max-programs $N --out $R/ctx --policy ctx --ctx-policy learn --ctx-mode learn --learner-cores $LC > $R/ctx/run.log 2>&1 || true
  echo "rl end \$(date)" >> \$LOG
fi
SER=()
[[ "$ARMS" == *revizor* ]] && SER+=(--series "Revizor=$R/revizor")
[[ "$ARMS" == *rl* ]] && SER+=(--series "RL (GRU)=$R/ctx")
$PY $T/../spectre_v1/plot_curve.py --out $T/figs/$NAME --no-grid --no-joins \\
    --title "$BASE ($CPU): cumulative violations vs programs" "\${SER[@]}" >> \$LOG 2>&1 || true
echo "done \$(date)" >> \$LOG
EOF
# the plot script counts Revizor's last -n program from a launch.sh in the run dir
printf '#!/bin/bash\n# (see ../launch.sh) revizor.py fuzz ... -n %s -i 50\n' "$N" > $R/revizor/launch.sh
chmod +x $R/launch.sh
if [ "${INLINE:-0}" = 1 ]; then
  echo "running: $R"
  $R/launch.sh
else
  (cd $R && setsid nohup ./launch.sh > launch.out 2>&1 < /dev/null &)
  echo "started: $R"
  echo "progress: tail $R/launch.log; machine info: $R/machine.txt"
fi
