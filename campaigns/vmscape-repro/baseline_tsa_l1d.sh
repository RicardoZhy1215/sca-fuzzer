#!/usr/bin/env bash
#
# baseline_tsa_l1d.sh — reproduce the PAPER's TSA-L1D as shipped: BOTH guests at KERNEL privilege
# (demo/tsa-l1d/config.yaml + template.asm), UNMODIFIED. This is the baseline our guest-user
# variants (config-fuzz-u2u/u2k) deviated from and were never checked against.
#
# Purpose: does this box (AMD Zen4, microcode 0206) detect the KNOWN both-kernel TSA-L1D?
#   - violation found  -> the pipeline + box reproduce real TSA-L1D; use as the reference.
#   - no violation     -> our guest-user results need re-examination before any TSA attribution.
#
# Runs the demo EXACTLY as published (its own config: seeds 20000000/1000000, threshold 0.1,
# both guests kernel, read-only observer page, hpa_gpa on). Needs the clean module + perf_env.sh.
#
# Usage:
#   bash baseline_tsa_l1d.sh                 # N=10000 x I=50 (demo default scale)
#   N=50000 bash baseline_tsa_l1d.sh
#   SMOKE=1 bash baseline_tsa_l1d.sh         # 10-TC pipeline check
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
DEMO_CONFIG=$REPO/demo/tsa-l1d/config.yaml  # shipped demo, both guests kernel
CONFIG=$CAMP/config-baseline-tsa-l1d.yaml   # derived: same config, threshold 0.5
TEMPLATE=$REPO/demo/tsa-l1d/template.asm
N=${N:-10000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=10; I=10; }
OUT=$CAMP/out-baseline-tsa-l1d
NONSTOP=${NONSTOP:-1}; NS_FLAG=(); [ "$NONSTOP" = "1" ] && NS_FLAG=(--nonstop)

[ -f "$DEMO_CONFIG" ] || { echo "missing $DEMO_CONFIG"; exit 2; }
# derive from the shipped demo, changing ONLY the threshold to 0.5 (both guests stay kernel)
sed -e "s/^analyser_stat_threshold:.*/analyser_stat_threshold: 0.5/" "$DEMO_CONFIG" > "$CONFIG"
[ -e /sys/rvzr_executor ] || { echo "[mod] module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

echo "=================================================================="
echo " BASELINE — real TSA-L1D (BOTH guests KERNEL, threshold 0.5)"
echo "   config   : $CONFIG  (derived from demo/tsa-l1d, thr $(grep -E '^analyser_stat_threshold' "$CONFIG" | awk '{print $2}'))"
echo "   privilege: $(grep -A1 'vm:' "$CONFIG" | grep privilege || true) (observer) / victim kernel"
echo "   template : $TEMPLATE"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)"
echo "   module   : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
echo "   env      : gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)"
echo "=================================================================="

# --- hpa_gpa EPT-collision pre-check (a colliding layout fails every load -> false 'no violation') ---
echo "[precheck] verifying the layout can map the sandbox..."
b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
tmp=$(mktemp -d); cd "$REPO"
timeout 90 "$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n 2 -i 10 -w "$tmp" >/dev/null 2>&1 || true
a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
rm -rf "$tmp"
if [ "$a" -gt "$b" ]; then
    echo "[precheck] !! hpa_gpa EPT COLLISION — every load fails. Fix: bash $CAMP/reload_until_ok.sh"; exit 3
fi
echo "[precheck] OK — no collision."

mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S)) — live log:"
echo "------------------------------------------------------------------"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT" "${NS_FLAG[@]}"
RC=$?
echo "------------------------------------------------------------------"
shopt -s nullglob
Vs=( "$OUT"/violation-* )
echo " violations: ${#Vs[@]}   (in $OUT)"
if [ "${#Vs[@]}" -gt 0 ]; then
    echo " => BASELINE CONFIRMED: this box reproduces real (both-kernel) TSA-L1D."
    echo "    Latest: $(ls -dt "$OUT"/violation-* | head -1)"
    echo "    Now the guest-user result can be compared against this reference."
else
    echo " => NO both-kernel TSA-L1D at N=$N. If our guest-user config leaks but this does NOT,"
    echo "    the guest-user leak is NOT plain TSA-L1D — it depends on the user/supervisor fault."
    echo "    (Raise N before concluding; the demo seed is fixed so more N = deeper deterministic walk.)"
fi
echo "=================================================================="
exit "$RC"
