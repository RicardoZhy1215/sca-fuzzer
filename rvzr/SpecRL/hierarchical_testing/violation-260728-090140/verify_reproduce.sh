#!/usr/bin/env bash
#
# verify_reproduce.sh — reproduce violation-260728-090140 in a QUIESCED environment
# to decide whether it is a real ct+DEH (write-#PF / Meltdown-family) leak or an
# SMT false positive.
#
# The training run had "WARNING: SMT is on! You may experience false positives",
# and the counterexample htraces differ by only ONE cache line — a classic marginal
# signal. This script kills the two biggest noise sources (SMT + NMI watchdog),
# raises the executor sample sizes / repetitions for statistical power, and re-runs
# the exact saved test case + inputs through `revizor.py reproduce`.
#
# Verdict:
#   * violation REPRODUCES with SMT off  -> real (proceed to minimize).
#   * violation DISAPPEARS with SMT off  -> it was SMT noise (false positive).
#
# The SMT/NMI toggles need root (sudo); the revizor run itself does not
# (the /sys/rvzr_executor sysfs is world-writable and the code runs as $USER).
set -uo pipefail

# ---- paths -----------------------------------------------------------------
REPO="/home/mluo/sca-fuzzer"
VDIR="$REPO/rvzr/SpecRL/hierarchical_testing/violation-260728-090140"
BASE_JSON="$REPO/base.json"
PY="/home/mluo/miniconda3/envs/specrl/bin/python"       # the env revizor runs under
PROGRAM="$VDIR/program.asm"
SRC_CONF="$VDIR/reproduce.yaml"                          # faithful config (SSBD on, seed 210)
VERIFY_CONF="$VDIR/reproduce_verify.yaml"                # = SRC_CONF + more samples
LOG="$VDIR/verify_reproduce.log"

# ---- confidence knobs (bigger than defaults [10,50,100,500] / 10) ----------
SAMPLE_SIZES="[100, 500, 1000, 5000]"
REPETITIONS=50

echo "=================================================================="
echo " Reproduce verification for violation-260728-090140"
echo " Contract: ct + delayed-exception-handling | SSBD ON | writable:false"
echo "=================================================================="

# ---- 1. quiesce the machine (root) -----------------------------------------
PRIOR_SMT="$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo unknown)"
PRIOR_NMI="$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null || echo unknown)"
echo "[env] prior SMT=$PRIOR_SMT  nmi_watchdog=$PRIOR_NMI"

echo "[env] disabling SMT and NMI watchdog (needs sudo)..."
echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null
echo 0   | sudo tee /proc/sys/kernel/nmi_watchdog        >/dev/null
# best-effort: pin frequency to reduce scaling noise (ignore if cpufreq absent)
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -w "$g" ] && echo performance | sudo tee "$g" >/dev/null 2>&1
done

NOW_SMT="$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo unknown)"
echo "[env] SMT now = $NOW_SMT   (want: off / forceoff)"
if [ "$NOW_SMT" = "on" ]; then
    echo "[env] WARNING: SMT is still ON — could not disable it. Results may be noisy."
fi

# ---- 2. build the high-confidence reproduce config -------------------------
cp "$SRC_CONF" "$VERIFY_CONF"
{
    echo ""
    echo "# --- appended by verify_reproduce.sh: raise statistical power ---"
    echo "executor_sample_sizes: $SAMPLE_SIZES"
    echo "executor_repetitions: $REPETITIONS"
} >> "$VERIFY_CONF"
echo "[conf] wrote $VERIFY_CONF (sample_sizes=$SAMPLE_SIZES, repetitions=$REPETITIONS)"

# ---- 3. collect inputs in fuzzing order (0000..0099) -----------------------
# -i takes a FILE LIST (not a dir); the glob sorts lexicographically -> in order,
# so counterexample indices (#16, #66) line up with the original run.
mapfile -t INPUTS < <(ls "$VDIR"/input_*.bin | sort)
echo "[inputs] ${#INPUTS[@]} input files"
if [ "${#INPUTS[@]}" -eq 0 ]; then
    echo "ERROR: no input_*.bin found in $VDIR" ; exit 2
fi

# ---- 4. run reproduce ------------------------------------------------------
echo "[run] revizor.py reproduce ... (log -> $LOG)"
cd "$REPO" || exit 2
set -x
"$PY" "$REPO/revizor.py" reproduce \
    -s "$BASE_JSON" \
    -c "$VERIFY_CONF" \
    -t "$PROGRAM" \
    -i "${INPUTS[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set +x

# ---- 5. verdict ------------------------------------------------------------
echo "=================================================================="
if grep -qiE '(^|[^a-z])violation' "$LOG" && grep -qiE 'Counterexample|Contract trace' "$LOG"; then
    HIT=1
else
    HIT=0
fi
echo " reproduce exit code = $RC   (nonzero usually = violation found)"
if [ "$RC" -ne 0 ] || [ "$HIT" -eq 1 ]; then
    echo " RESULT: violation REPRODUCED with SMT off  ->  looks REAL."
    echo "         Next step: minimize with $VDIR/minimize.yaml to pin the mechanism."
else
    echo " RESULT: NO violation with SMT off  ->  most likely an SMT FALSE POSITIVE."
fi
echo "=================================================================="
echo "[env] SMT is still OFF. To restore when you are done verifying:"
echo "        echo ${PRIOR_SMT:-on} | sudo tee /sys/devices/system/cpu/smt/control"
echo "        echo ${PRIOR_NMI:-1}  | sudo tee /proc/sys/kernel/nmi_watchdog"
