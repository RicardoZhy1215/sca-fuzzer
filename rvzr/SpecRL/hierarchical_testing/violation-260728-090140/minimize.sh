#!/usr/bin/env bash
#
# minimize.sh — shrink violation-260728-090140 to its essential instructions to
# pin the mechanism (which faulting store + which leaking load produce the
# cache-line-9 leak under ct + delayed-exception-handling).
#
# MUST run with SMT off: minimize repeatedly re-checks "is the violation still
# there?", so measurement noise would make it delete needed instructions. The
# verify_reproduce.sh run already left SMT off; this script re-asserts it (idempotent).
#
# Passes enabled (see `revizor.py minimize -h`):
#   instruction   (default) iteratively delete instructions that aren't needed
#   label         (default) drop unused labels
#   simplification          replace complex insns with simpler equivalents
#   constant                replace constants with 0 where possible
#   mask                    shrink instrumentation masks
#   comment                 annotate the asm with which insns cause the violation
# (fence + input-diff are left for a focused follow-up; see notes at the end.)
set -uo pipefail

# ---- paths -----------------------------------------------------------------
REPO="/home/mluo/sca-fuzzer"
VDIR="$REPO/rvzr/SpecRL/hierarchical_testing/violation-260728-090140"
BASE_JSON="$REPO/base.json"
PY="/home/mluo/miniconda3/envs/specrl/bin/python"
PROGRAM="$VDIR/program.asm"
SRC_CONF="$VDIR/minimize.yaml"                     # seed 210, inputs_per_class 2 (discovery config)
MIN_CONF="$VDIR/minimize_verify.yaml"             # = SRC_CONF + more samples
OUT_ASM="$VDIR/minimized.asm"
OUT_INPUTS="$VDIR/minimized_inputs"
LOG="$VDIR/minimize.log"

# ---- confidence knobs (strong signal here: 6246 vs 0, but keep headroom) ----
SAMPLE_SIZES="[100, 500, 1000]"
REPETITIONS=20
NUM_INPUTS=100          # matches the discovery run ("Inputs per test case: 100")
NUM_ATTEMPTS=2

echo "=================================================================="
echo " Minimizing violation-260728-090140  (ct + DEH, write-#PF family)"
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
[ "$SMT" = "on" ] && echo "[env] WARNING: SMT still ON — minimize decisions may be noisy."

# ---- 2. build the minimize config -----------------------------------------
cp "$SRC_CONF" "$MIN_CONF"
{
    echo ""
    echo "# --- appended by minimize.sh ---"
    echo "executor_sample_sizes: $SAMPLE_SIZES"
    echo "executor_repetitions: $REPETITIONS"
} >> "$MIN_CONF"
echo "[conf] $MIN_CONF (sample_sizes=$SAMPLE_SIZES, repetitions=$REPETITIONS)"

# ---- 3. run minimize -------------------------------------------------------
mkdir -p "$OUT_INPUTS"
echo "[run] minimizing (this can take several minutes; log -> $LOG)"
cd "$REPO" || exit 2
set -x
"$PY" "$REPO/revizor.py" minimize \
    -s "$BASE_JSON" \
    -c "$MIN_CONF" \
    -t "$PROGRAM" \
    -i "$NUM_INPUTS" \
    -o "$OUT_ASM" \
    --input-outdir "$OUT_INPUTS" \
    --num-attempts "$NUM_ATTEMPTS" \
    --enable-instruction-pass    true \
    --enable-label-pass          true \
    --enable-simplification-pass true \
    --enable-constant-pass       true \
    --enable-mask-pass           true \
    --enable-comment-pass        true 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set +x

# ---- 4. show the result ----------------------------------------------------
echo "=================================================================="
echo " minimize exit code = $RC"
if [ -f "$OUT_ASM" ]; then
    echo " Minimized test case -> $OUT_ASM"
    echo "------------------------------------------------------------------"
    grep -vE '^\s*$' "$OUT_ASM"
    echo "------------------------------------------------------------------"
    echo " (Lines annotated by the comment pass mark the violation-relevant insns.)"
else
    echo " No minimized.asm produced — check $LOG."
fi
echo "=================================================================="
echo " Follow-ups on the minimized program (run manually if wanted):"
echo "   * fence pass  — locate the speculative window:"
echo "       ...minimize ... -t minimized.asm --enable-instruction-pass false \\"
echo "          --enable-fence-pass true -o minimized_fenced.asm"
echo "   * input-diff  — find which secret bytes matter:"
echo "       ...minimize ... -t minimized.asm --enable-instruction-pass false \\"
echo "          --enable-input-diff-pass true --input-outdir min_inputs_diff -o /dev/null"
echo " SMT is still OFF — restore when fully done:"
echo "       echo on | sudo tee /sys/devices/system/cpu/smt/control"
echo "       echo 1  | sudo tee /proc/sys/kernel/nmi_watchdog"
