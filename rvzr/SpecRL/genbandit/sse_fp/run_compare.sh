#!/bin/bash
# SSE-FP: unmodified Revizor, then ctxrl (GRU, learned modes, xmm families), N programs each, on the
# same config; then the cumulative-violations figure. CPU-agnostic (Revizor picks the Intel or AMD
# executor from /proc/cpuinfo). Runs detached: closing the session does not stop it.
#
#   cd <sca-fuzzer checkout>
#   PY=<python with Revizor's deps + torch> rvzr/SpecRL/genbandit/sse_fp/run_compare.sh <entropy 31|16> <N> [fuzzer cores] [learner cores]
#   e.g.  rvzr/SpecRL/genbandit/sse_fp/run_compare.sh 31 20000            # cores 1-3 / 4-7
#         rvzr/SpecRL/genbandit/sse_fp/run_compare.sh 16 20000 0-1 2-5    # fewer cores
# On a hybrid Intel CPU (P- and E-cores) give P-cores only: E-cores are a different microarchitecture.
# Output: sse_fp/runs/<host>-<cpu>-e<entropy>-<N>-<date>/{revizor,ctx}/, figure sse_fp/figs/<same name>.*
set -eu
E=${1:?entropy 31 or 16}; N=${2:?programs per tool}; FC=${3:-1-3}; LC=${4:-4-7}
ROOT=$(git rev-parse --show-toplevel)
S=$ROOT/rvzr/SpecRL/genbandit/sse_fp
PY=$(readlink -f "${PY:-$(command -v python)}")
CPU=$(grep -m1 "model name" /proc/cpuinfo | sed 's/.*: //; s/[^A-Za-z0-9]\+/_/g' | cut -c1-40)
NAME=$(hostname -s)-$CPU-e$E-$N-$(date +%y%m%d-%H%M)
R=$S/runs/$NAME
mkdir -p $R/revizor $R/ctx $S/figs
CFG=$("$PY" -c "import sys; sys.path.insert(0, '$S/..'); sys.path.insert(0, '$S'); import space_ssefp as s; print(s.write({**s.CONTROL, 'entropy_bits': $E}))")
cp "$CFG" $R/revizor/config.yaml
cat > $R/launch.sh <<EOF
#!/bin/bash
cd $ROOT
LOG=$R/launch.log
{ echo "host \$(hostname)"; grep -m1 "model name" /proc/cpuinfo; grep -m1 microcode /proc/cpuinfo; uname -r
  echo "smt \$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)"; cat /proc/cmdline; git rev-parse HEAD; } > $R/machine.txt
exec 9>$S/.executor.lock
flock 9
while pgrep -f "^[^ ]*python[^ ]* ([^ ]*/)?(revizor\.py fuzz|tcenv\.py|ttk_bench\.py)" > /dev/null; do sleep 30; done
echo "revizor start \$(date)" >> \$LOG
taskset -c $FC $PY revizor.py fuzz -s base.json -c $R/revizor/config.yaml -I $R/revizor -n $N -i 50 -w $R/revizor --nonstop > $R/revizor/fuzz.log 2>&1
echo "revizor end \$(date) rc \$?" >> \$LOG
echo "rl start \$(date)" >> \$LOG
taskset -c $FC $PY $S/../tcenv.py --space ssefp --action '{"entropy_bits": $E}' --mode bypass --minutes 2880 \\
    --max-programs $N --out $R/ctx --policy ctx --ctx-policy learn --ctx-mode learn --learner-cores $LC > $R/ctx/run.log 2>&1
echo "rl end \$(date) rc \$?" >> \$LOG
$PY $S/../spectre_v1/plot_curve.py --out $S/figs/$NAME --no-grid --no-joins \\
    --title "SSE floating point ($CPU, entropy $E): cumulative violations vs programs" \\
    --series "Revizor=$R/revizor" --series "RL (GRU)=$R/ctx" >> \$LOG 2>&1
echo "done \$(date)" >> \$LOG
EOF
# the plot script counts Revizor's last -n program from a launch.sh in the run dir
printf '#!/bin/bash\n# (see ../launch.sh) revizor.py fuzz ... -n %s -i 50\n' "$N" > $R/revizor/launch.sh
chmod +x $R/launch.sh
(cd $R && setsid nohup ./launch.sh > launch.out 2>&1 < /dev/null &)
echo "started: $R"
echo "progress: tail $R/launch.log; machine info: $R/machine.txt"
