#!/usr/bin/env bash
#
# refuzz_u2u.sh — a FRESH fuzzing campaign on the guest-USER -> guest-USER config, to find NEW
# cross-actor violations.
#
# Uses a RANDOM program/data seed (seed=0), so every run walks a fresh set of test cases the first
# campaign never saw. Fast fuzzing config (fast-path on, small samples) for throughput.
#
# Needs the clean module (1CD139EF) in a NON-colliding layout + perf_env.sh. The script pre-checks
# the hpa_gpa EPT collision (a colliding layout makes EVERY test-case load fail).
#
# Usage:
#   bash refuzz_u2u.sh                 # N=50000 x I=50, random seed
#   N=100000 bash refuzz_u2u.sh        # longer sweep
#   SMOKE=1 bash refuzz_u2u.sh         # 10-TC pipeline check
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
TEMPLATE=$CAMP/fuzz-multiactor.asm
N=${N:-50000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=10; I=10; }
OUT=$CAMP/out-refuzz-u2u
CONFIG=$CAMP/config-refuzz-u2u.yaml
# keep fuzzing after a violation (store them all) — set NONSTOP=0 to stop at the first
NONSTOP=${NONSTOP:-1}; NS_FLAG=(); [ "$NONSTOP" = "1" ] && NS_FLAG=(--nonstop)

# seed = 0 -> Revizor picks a RANDOM program/data seed each run (fuzzing needs no fixed seed; it logs
# the chosen seed, and violations are saved as artifacts so they're reproducible regardless).
sed -e "s/^program_generator_seed:.*/program_generator_seed: 0/" \
    -e "s/^data_generator_seed:.*/data_generator_seed: 0/" \
    -e "s/^analyser_stat_threshold:.*/analyser_stat_threshold: 0.5/" \
    "$CAMP/config-fuzz-u2u.yaml" > "$CONFIG"

echo "=================================================================="
echo " RE-FUZZ guest-USER -> guest-USER (find NEW violations)"
echo "   template : $TEMPLATE"
echo "   config   : $CONFIG   (program/data seed = 0 -> random each run)"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)"
echo "=================================================================="

[ -e /sys/rvzr_executor ] || { echo "[mod] module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
echo "[mod] loaded: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
echo "[env] governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)  (want performance/off; run perf_env.sh)"



mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S)) — live log:"
echo "------------------------------------------------------------------"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT" "${NS_FLAG[@]}"
RC=$?
echo "------------------------------------------------------------------"
shopt -s nullglob
Vs=( "$OUT"/violation-* ); Bs=( "$OUT"/bugs/violation-* )
echo " violations: ${#Vs[@]}   arch-mismatch bugs: ${#Bs[@]}   (in $OUT)"
if [ "${#Vs[@]}" -gt 0 ]; then
    echo " => NEW candidate cross-actor leak(s). Latest: $(ls -dt "$OUT"/violation-* | head -1)"
    echo "    triage: 0 jcc/indirect => fault-driven; check the observer's input is byte-identical"
    echo "    across the counterexample; replay/verify with replay_fuzz.sh (V=<that dir>)."
else
    echo " => no violation at N=$N. Raise N, or just re-run (random seed each time -> new programs)."
fi
echo "=================================================================="
exit "$RC"
