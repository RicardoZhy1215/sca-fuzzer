#!/usr/bin/env bash
#
# reproduce_minimized.sh — confirm the MINIMIZED test case still triggers the
# violation (ct + delayed-exception-handling, write-#PF / Meltdown-family).
#
# Why no -i input files: minimize did not save its inputs (input-diff/seq passes
# were off), and the boosted inputs are TAINT-driven — they depend on the program.
# minimized.asm has a different taint than the original program.asm, so the old
# input_*.bin are NOT the right inputs for it. Instead we omit -i, which makes
# fuzzer.start() regenerate `-n` base inputs from the config seed (210) and re-boost
# them for minimized.asm inside fuzzing_round — exactly what minimize did to verify.
#
# Run with SMT off (verify_reproduce.sh already left it off; re-asserted below).
set -uo pipefail

# ---- paths -----------------------------------------------------------------
REPO="/home/mluo/sca-fuzzer"
VDIR="$REPO/rvzr/SpecRL/hierarchical_testing/violation-260728-090140"
BASE_JSON="$REPO/base.json"
PY="/home/mluo/miniconda3/envs/specrl/bin/python"
PROGRAM="$VDIR/minimized.asm"
SRC_CONF="$VDIR/minimize.yaml"                      # seed 210, inputs_per_class 2, writable:false, DEH, SSBD on
VERIFY_CONF="$VDIR/reproduce_min_verify.yaml"       # = SRC_CONF + more samples
LOG="$VDIR/reproduce_minimized.log"

# ---- confidence knobs ------------------------------------------------------
SAMPLE_SIZES="[100, 500, 1000, 5000]"
REPETITIONS=50
NUM_INPUTS=100          # matches the discovery/minimize run

echo "=================================================================="
echo " Reproducing MINIMIZED violation-260728-090140"
echo " -t minimized.asm | regenerate inputs from seed 210 (no -i) | SMT off"
echo "=================================================================="

# ---- 1. re-assert quiesced environment (only sudo if needed) ---------------
SMT="$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo unknown)"
if [ "$SMT" = "on" ]; then
    echo "[env] SMT is on -> disabling (sudo)..."
    echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null 2>&1
    echo 0   | sudo tee /proc/sys/kernel/nmi_watchdog        >/dev/null 2>&1
    SMT="$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo unknown)"
fi
echo "[env] SMT = $SMT  (want off/forceoff)"
[ "$SMT" = "on" ] && echo "[env] WARNING: SMT still ON — results may be noisy."

# ---- 2. build the high-confidence config -----------------------------------
cp "$SRC_CONF" "$VERIFY_CONF"
{
    echo ""
    echo "# --- appended by reproduce_minimized.sh ---"
    echo "executor_sample_sizes: $SAMPLE_SIZES"
    echo "executor_repetitions: $REPETITIONS"
} >> "$VERIFY_CONF"
echo "[conf] $VERIFY_CONF (sample_sizes=$SAMPLE_SIZES, repetitions=$REPETITIONS)"

# ---- 3. run reproduce (no -i => regenerate + re-boost from seed 210) --------
echo "[run] revizor.py reproduce -t minimized.asm ... (log -> $LOG)"
cd "$REPO" || exit 2
set -x
"$PY" "$REPO/revizor.py" reproduce \
    -s "$BASE_JSON" \
    -c "$VERIFY_CONF" \
    -t "$PROGRAM" \
    -n "$NUM_INPUTS" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set +x

# ---- 4. verdict ------------------------------------------------------------
echo "=================================================================="
if grep -qiE 'Violations detected|Violations: [1-9]' "$LOG"; then HIT=1; else HIT=0; fi
echo " reproduce exit code = $RC"
if [ "$RC" -ne 0 ] || [ "$HIT" -eq 1 ]; then
    echo " RESULT: MINIMIZED test case STILL triggers the violation  ->  confirmed."
else
    echo " RESULT: minimized test case did NOT reproduce — check $LOG."
    echo "         (If it fails, the minimizer may have over-reduced; fall back to"
    echo "          program.asm + saved input_*.bin via verify_reproduce.sh.)"
fi
echo "=================================================================="
echo " SMT is still OFF — restore when fully done:"
echo "     echo on | sudo tee /sys/devices/system/cpu/smt/control"
echo "     echo 1  | sudo tee /proc/sys/kernel/nmi_watchdog"
