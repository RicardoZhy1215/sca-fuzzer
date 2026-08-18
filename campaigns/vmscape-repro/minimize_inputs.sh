#!/usr/bin/env bash
#
# minimize_inputs.sh — run ONLY the INPUT passes (diff + seq) to isolate the LEAKED VALUES.
#
# Instruction pass OFF (program unchanged). The input-diff-pass minimizes the byte-difference between
# the two counterexample inputs while the violation holds, so the bytes that SURVIVE are exactly the
# ones the leak depends on — i.e. the leaked values. input-seq-pass trims the input set to the
# minimal reproducing subset.
#
# vs the doc's example (-i 25, generate-from-seed): we LOAD the saved inputs via RVZR_MINIMIZE_INPUT_DIR
# and use -i = full count, so the [0,50] counterexample is present (the doc's -i 25 -> only 0-49).
#
# NB: the input passes validate each reduction against the (weak/drifting) violation — so run this
# in an UP-window (N=5 replay_fuzz_u2k.sh should be ~4-5/5), or it'll fail like a low-signal minimize.
#
# Usage:
#   bash minimize_inputs.sh                       # on program.min.asm (u2k), strong config
#   PROG=$V/program.asm bash minimize_inputs.sh   # on the original program instead
#   V=<dir> CONFIG=<yaml> bash minimize_inputs.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-fuzz-u2k/violation-260814-201415}
CONFIG=${CONFIG:-$CAMP/config-fuzz-u2k-strong.yaml}
PROG=${PROG:-$V/program.min.asm}          # program to run the input passes against
OUT=${OUT:-$V/program.inputmin.asm}       # -o is required; program is unchanged (instruction pass off)
INDIR=${INDIR:-$V/min-inputs}
INPUTS=${INPUTS:-$(ls "$V"/input_*.bin 2>/dev/null | wc -l)}
ATTEMPTS=${ATTEMPTS:-1}

[ -f "$PROG" ] || { echo "no program at $PROG (run minimize_violation.sh first, or set PROG=)"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

echo "=================================================================="
echo " MINIMIZE INPUTS (isolate leaked values) — input-diff + input-seq passes"
echo "   program : $PROG"
echo "   config  : $CONFIG"
echo "   inputs  : $INPUTS saved (loaded)   out-dir: $INDIR"
echo "   module  : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "=================================================================="
mkdir -p "$INDIR"
export RVZR_MINIMIZE_INPUT_DIR="$V"       # load the exact saved inputs (incl #50)
cd "$REPO"
"$PY" revizor.py minimize -s "$BASE" -c "$CONFIG" -I "$V" \
    -t "$PROG" -i "$INPUTS" -o "$OUT" --input-outdir "$INDIR" --num-attempts "$ATTEMPTS" \
    --enable-instruction-pass false \
    --enable-input-diff-pass true \
    --enable-input-seq-pass true
RC=$?
echo "------------------------------------------------------------------"
echo " minimized inputs written to $INDIR:"
ls -l "$INDIR"/*.bin 2>/dev/null | awk '{print "   "$NF}' | head
n=$(ls "$INDIR"/*.bin 2>/dev/null | wc -l)
if [ "$n" -ge 2 ]; then
    a=$(ls "$INDIR"/*.bin | sed -n 1p); b=$(ls "$INDIR"/*.bin | sed -n 2p)
    echo " === LEAKED VALUES: differing bytes between the two counterexample inputs ==="
    echo "   ($a  vs  $b)"
    cmp -l "$a" "$b" 2>/dev/null | awk '{printf "   byte %-6d : %o vs %o\n", $1-1, $2, $3}' | head -40
    echo "   (total differing bytes: $(cmp -l "$a" "$b" 2>/dev/null | wc -l))"
    echo "   [byte offsets map to the input layout: per-actor 12288B = main[0-12287] vmvictim[12288-24575] vm[24576-36863]]"
else
    echo " (no minimized inputs produced — PASS 1 likely didn't reproduce; run in an up-window)"
fi
echo "=================================================================="
exit "$RC"
