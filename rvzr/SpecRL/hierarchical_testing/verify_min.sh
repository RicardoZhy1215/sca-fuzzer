#!/usr/bin/env bash
# Verify that the MINIMIZED test case still triggers the violation.
#
# Runs `revizor reproduce` on min.asm with the minimized inputs (min-inputs/),
# using the folder's reproduce.yaml (ct + delayed-exception-handling, present:false).
# This is the documented post-minimization check, cf. docs/howto/minimize.md:183
#   rvzr reproduce -s base.json -c <viol>/reproduce.yaml -t min.asm -i ./inputs/min_input*.bin
#
# No sudo needed. Run it a few times (REPEAT=10) to judge stability: a genuine
# violation reproduces consistently; a sporadic one is likely a noise false positive
# (see docs/howto/interpret-results.md).
#
# Usage:
#   bash verify_min.sh                       # 1 run, default folder
#   REPEAT=10 bash verify_min.sh             # 10 runs, report how many reproduced
#   ASM=min.asm INDIR=min-inputs bash verify_min.sh <violation-folder>
set -u
REPO=/home/hz25d/beyond_v1_v4/sca-fuzzer
PY=/home/hz25d/miniconda3/envs/specrl/bin/python
HT=rvzr/SpecRL/hierarchical_testing
NAME="${1:-violation-260721-212748}"
ASM="${ASM:-min.asm}"
INDIR="${INDIR:-min-inputs}"
REPEAT="${REPEAT:-1}"
TIMEOUT="${TIMEOUT:-none}"      # 'none'/0 = unlimited

V="$HT/$NAME"
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }
for f in "$V/reproduce.yaml" "$V/$ASM" base.json; do
    [ -f "$f" ] || { echo "missing: $REPO/$f"; exit 1; }
done

shopt -s nullglob
inputs=( "$V/$INDIR"/min_input*.bin )
[ ${#inputs[@]} -eq 0 ] && inputs=( "$V/$INDIR"/input*.bin )
shopt -u nullglob
[ ${#inputs[@]} -gt 0 ] || { echo "no min_input*.bin in $REPO/$V/$INDIR"; exit 1; }

if [ "$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)" = "on" ]; then
    echo "WARNING: SMT is ON -> NMI/PMI can corrupt measurements (EIO). Disable it first:"
    echo "         echo off | sudo tee /sys/devices/system/cpu/smt/control"
fi
pgrep -af 'train\.py' >/dev/null 2>&1 && echo "WARNING: train.py is running -- it competes for the executor KM."

TO=(timeout --signal=INT "$TIMEOUT")
{ [ "$TIMEOUT" = "0" ] || [ "$TIMEOUT" = "none" ]; } && TO=()

echo "folder   : $NAME"
echo "program  : $V/$ASM   ($(grep -cvE '^[[:space:]]*#|^[[:space:]]*$|^\.' "$V/$ASM") instructions)"
echo "inputs   : ${#inputs[@]} from $V/$INDIR"
echo "config   : $V/reproduce.yaml (ct + delayed-exception-handling)"
echo "repeat   : $REPEAT"
echo "------------------------------------------------------------"

yes_n=0; no_n=0; err_n=0
for i in $(seq 1 "$REPEAT"); do
    log="$V/verify_min_run${i}.log"
    "${TO[@]}" "$PY" ./revizor.py reproduce \
        -s base.json \
        -c "$V/reproduce.yaml" \
        -t "$V/$ASM" \
        -i "${inputs[@]}" \
        > "$log" 2>&1
    rc=$?

    if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then r="TIMEOUT"; err_n=$((err_n+1))
    elif grep -aqiE 'Input/output error|CalledProcessError|Traceback' "$log"; then r="ERROR(EIO/NMI)"; err_n=$((err_n+1))
    elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then r="NO"; no_n=$((no_n+1))
    elif grep -aqiE 'violation' "$log"; then r="YES"; yes_n=$((yes_n+1))
    else r="?"; err_n=$((err_n+1)); fi
    printf "run %2d/%s : %-14s (rc=%s, log=%s)\n" "$i" "$REPEAT" "$r" "$rc" "$log"
done

echo "------------------------------------------------------------"
echo "RESULT: reproduced(YES)=$yes_n   not-reproduced(NO)=$no_n   error/timeout=$err_n   of $REPEAT"
if [ "$err_n" -gt 0 ]; then
    echo "note: EIO/NMI errors are measurement failures, not verdicts -- disable SMT and re-run those."
fi
