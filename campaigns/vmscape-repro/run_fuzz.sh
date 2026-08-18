#!/usr/bin/env bash
#
# run_fuzz.sh [u2u|u2k] — FUZZ guest-user actor configs for cross-actor speculative leaks.
#
# The Enter/Exit Revizor paper didn't cover guest-user actors; this fuzzes the two cases the
# guest-user extension enables, using the multi-actor fuzzing template (fuzz-multiactor.asm,
# derived from demo/tsa-l1d) — random_instructions in both guest bodies, ct-ni contract.
#   u2u : guest USER (attacker/vm) vs guest USER (victim/vmvictim)   -> config-fuzz-u2u.yaml
#   u2k : guest USER (attacker/vm) vs guest KERNEL (victim/vmvictim) -> config-fuzz-u2k.yaml
#
# Use the CLEAN executor build (1CD139EF) to find leaks — the IBPB build (A07CAEF9) flushes the BTB
# and will suppress branch-based cross-actor leaks. The script warns if IBPB is loaded.
# Load the clean build and apply perf_env.sh first (F+R needs the low-latency env):
#   sudo rmmod rvzr_executor; sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko   # want 1CD139EF
#
# Usage:
#   bash run_fuzz.sh u2u                # N=5000 x I=50
#   N=50000 bash run_fuzz.sh u2k        # longer campaign
#   SMOKE=1 bash run_fuzz.sh u2u        # 10-TC pipeline check
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
TEMPLATE=$CAMP/fuzz-multiactor.asm
MODE=${1:-${MODE:-u2u}}
case "$MODE" in
  u2u) CONFIG=$CAMP/config-fuzz-u2u.yaml; DESC="guest USER (vm) -> guest USER (vmvictim)";;
  u2k) CONFIG=$CAMP/config-fuzz-u2k.yaml; DESC="guest USER attacker (vm) -> guest KERNEL victim (vmvictim)";;
  *) echo "usage: run_fuzz.sh [u2u|u2k]"; exit 2;;
esac
OUT=$CAMP/out-fuzz-$MODE
N=${N:-5000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=10; I=10; }

echo "=================================================================="
echo " FUZZ cross-actor leaks: $DESC"
echo "   template : $TEMPLATE"
echo "   config   : $CONFIG   (ct-ni, F+R + hpa_gpa)"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)"
echo "=================================================================="

[ -e /sys/rvzr_executor ] || { echo "[mod] module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
SV=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)
echo "[mod] loaded: $SV"
[ "$SV" = "A07CAEF9A1C9FB1F6B5AD66" ] && echo "[mod] WARNING: this is the IBPB build — it will SUPPRESS branch-based leaks. Load clean 1CD139EF to fuzz."
echo "[env] governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)  (want performance/off; run perf_env.sh)"

mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S)) — live log:"
echo "------------------------------------------------------------------"
cd "$REPO"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT"
RC=$?
echo "------------------------------------------------------------------"
shopt -s nullglob
V=( "$OUT"/violation-* ); B=( "$OUT"/bugs/violation-* )
echo " violations: ${#V[@]}   arch-mismatch bugs: ${#B[@]}   (in $OUT)"
if [ "${#V[@]}" -gt 0 ]; then
    echo " => candidate cross-actor leak in the $MODE config. Latest: $(ls -dt "$OUT"/violation-* | head -1)"
    echo "    Triage: is it BTB (indirect jmps) or v1 (jcc)? does the htrace track vmvictim's input?"
    echo "    For a BTB hit, the trainer-strip tool confirms causality: V=<dir> bash reproduce_trainer_test.sh"
else
    echo " => no violation at N=$N. Increase N for a longer campaign, or lower analyser_stat_threshold."
fi
echo "=================================================================="
exit "$RC"
