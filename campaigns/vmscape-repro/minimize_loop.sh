#!/usr/bin/env bash
#
# minimize_loop.sh — retry minimize until PASS 1 catches an up-moment, then let that run finish.
#
# The violation is weak/drifting (~1/5), so minimize's PASS 1 ("Reproducing the violation") usually
# fails and exits fast. This loop relaunches minimize; each failed attempt costs ~10-30s; the first
# attempt whose PASS 1 reproduces proceeds through the full (slow) minimization and the loop stops.
#
# Detection: PASS-1 failure prints "Could not reproduce the violation. Exiting". If that string is
# NOT in the run's output, PASS 1 succeeded and minimize ran to completion -> done.
#
# Same knobs as minimize_violation.sh. To run the INPUT passes instead of program passes (isolate
# leaked values on the minimized program), set:  INPUT_MODE=1 PROG=$V/program.min.asm
#
# Usage:
#   bash minimize_loop.sh                       # retry program-minimize of u2k until it starts
#   MAX=60 bash minimize_loop.sh
#   INPUT_MODE=1 bash minimize_loop.sh          # retry the input-diff/seq passes instead
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-fuzz-u2k/violation-260814-201415}
CONFIG=${CONFIG:-$CAMP/config-fuzz-u2k.yaml}
INDIR=${INDIR:-$V/min-inputs}
INPUTS=${INPUTS:-$(ls "$V"/input_*.bin 2>/dev/null | wc -l)}
ATTEMPTS=${ATTEMPTS:-3}
MAX=${MAX:-40}
INPUT_MODE=${INPUT_MODE:-0}

if [ "$INPUT_MODE" = "1" ]; then
    PROG=${PROG:-$V/program.min.asm}; OUT=${OUT:-$V/program.inputmin.asm}
    FLAGS="--enable-instruction-pass false --enable-input-diff-pass true --enable-input-seq-pass true"
    MODE="INPUT passes (diff+seq) on $(basename "$PROG")"
else
    PROG=${PROG:-$V/program.asm}; OUT=${OUT:-$V/program.min.asm}
    FLAGS="--enable-simplification-pass true --enable-comment-pass true"
    MODE="program passes (instruction+simplification) on $(basename "$PROG")"
fi

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

echo "=================================================================="
echo " MINIMIZE LOOP — retry until PASS 1 reproduces, then finish"
echo "   mode    : $MODE"
echo "   config  : $CONFIG   inputs: $INPUTS (loaded)   max attempts: $MAX"
echo "=================================================================="
export RVZR_MINIMIZE_INPUT_DIR="$V"
cd "$REPO"
log=$(mktemp)
for a in $(seq 1 "$MAX"); do
    echo "----- attempt $a/$MAX  $(date +%H:%M:%S) -----"
    # tee to the log AND stream a filtered view live (so a slow successful run shows progress)
    "$PY" revizor.py minimize -s "$BASE" -c "$CONFIG" -I "$V" \
        -t "$PROG" -i "$INPUTS" -o "$OUT" --input-outdir "$INDIR" --num-attempts "$ATTEMPTS" $FLAGS 2>&1 \
        | tee "$log" \
        | grep --line-buffered -iE "PASS|reproduc|minimized|Could not|Violating|removed" | sed 's/^/    /' || true
    if grep -q "Could not reproduce the violation. Exiting" "$log"; then
        echo "    -> PASS 1 missed (drift) — retrying"
        continue
    fi
    echo "=================================================================="
    echo " *** PASS 1 hit on attempt $a — minimization completed ***  -> $OUT"
    rm -f "$log"
    exit 0
done
rm -f "$log"
echo "=================================================================="
echo " PASS 1 never reproduced in $MAX attempts — signal too low. Raise MAX, or wait for an up-window"
echo " (N=5 bash $CAMP/replay_fuzz_u2k.sh should show ~4-5/5)."
echo "=================================================================="
exit 1
