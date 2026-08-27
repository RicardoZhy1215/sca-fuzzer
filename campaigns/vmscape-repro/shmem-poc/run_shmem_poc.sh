#!/usr/bin/env bash
#
# run_shmem_poc.sh — run the MANUAL shared-memory transient-leak prototype and show the readout.
#
#   The template is a FIXED hand-written gadget (no random_instructions), so tfuzz runs the same
#   program across random INPUTS; the ct-ni detector flags a violation iff the observer's htrace
#   depends on the victim's secret S -> i.e. the transient read of the SHARED byte at 0x140 leaks.
#   A hit prints the "Violations detected" table; the DIFFERING probe line = (S & 0x3f) for the two
#   counterexample inputs -> the leaked value, read straight out of the htrace.
#
# Usage:
#   bash run_shmem_poc.sh              # N=5 x I=50 (fixed gadget; a few input sweeps)
#   SMOKE=1 bash run_shmem_poc.sh      # N=2 pipeline check
#   N=50 bash run_shmem_poc.sh         # more input coverage if it doesn't fire immediately
#   THR=0.3 bash run_shmem_poc.sh      # loosen threshold if the signal is borderline
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
HERE=/home/mluo/sca-fuzzer/campaigns/vmscape-repro/shmem-poc
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
TEMPLATE=$HERE/template-shmem-poc.asm
SRC_CONFIG=$HERE/config-shmem-poc.yaml
N=${N:-5}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=2; I=20; }
OUT=$HERE/out-shmem-poc

CONFIG=$SRC_CONFIG
if [ -n "${THR:-}" ]; then
    CONFIG=$HERE/config-shmem-poc.thr.yaml
    sed "s/^analyser_stat_threshold:.*/analyser_stat_threshold: $THR/" "$SRC_CONFIG" > "$CONFIG"
fi

echo "=================================================================="
echo " SHARED-MEMORY transient-leak PROTOTYPE (hand-written gadget)"
echo "   victim writes secret S -> shared 0x140 ; observer transiently reads it -> probe line (S&0x3f)"
echo "   template : $TEMPLATE   (fixed gadget, no random_instructions)"
echo "   config   : $CONFIG   (both guests kernel, hpa_gpa=true => shared main page, thr $(grep -E '^analyser_stat_threshold' "$CONFIG" | awk '{print $2}'))"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)"
echo "=================================================================="
[ -f "$TEMPLATE" ] || { echo "no template at $TEMPLATE"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "[mod] module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
echo "[mod] loaded: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
echo "[ucode] $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)   [env] gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)"

mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S)):"
echo "------------------------------------------------------------------"
cd "$REPO"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT"
RC=$?
echo "------------------------------------------------------------------"
shopt -s nullglob
Vs=( "$OUT"/violation-* )
echo " violations: ${#Vs[@]}   (in $OUT)"
if [ "${#Vs[@]}" -gt 0 ]; then
    LATEST=$(ls -dt "$OUT"/violation-* | head -1)
    echo " => SHARED-MEM TRANSIENT EXFIL WORKS. Latest: $LATEST"
    echo "    The observer learned the victim's shared byte 0x140 via the fault-shadow read."
    echo "    Reproduce + dump the htrace to read the value out: the DIFFERING probe line = (S & 0x3f)."
    echo "    Next (hard leak): block vm's arch read of 0x140 (present:false region) + microtag path."
else
    echo " => no violation. Either the transient read/encode didn't fire (try N=50, THR=0.3, perf_env.sh),"
    echo "    or the pipeline is fine but the gadget needs tuning (see the fault-shadow block in the"
    echo "    template — the store must fault and the 0x140 read must be in its shadow)."
fi
echo "=================================================================="
exit "$RC"
