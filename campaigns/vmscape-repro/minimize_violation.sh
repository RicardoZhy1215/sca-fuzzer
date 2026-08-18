#!/usr/bin/env bash
#
# minimize_violation.sh — shrink a saved violating test case to its ESSENTIAL leaking gadget,
# using Revizor's `minimize` subcommand. Reveals which of the ~64+64 fuzzed instructions actually
# cause the leak (huge for understanding a fuzzing find).
#
# Passes (Revizor docs):
#   instruction  (ON by default)  — iteratively REMOVE instructions while the violation holds
#   simplification (SIMPLIFY=1)    — replace complex instrs with simpler ones
#   nop (NOP=1) / constant (CONSTANT=1) / mask (MASK=1) / fence (FENCE=1) / comment (COMMENT=1)
#
# IMPORTANT: minimize normally GENERATES inputs from the config seed, which fails for a long fuzzing
# campaign; so this script LOADS the exact saved inputs via RVZR_MINIMIZE_INPUT_DIR (see below).
#
# SLOW: O(#instructions x one full violation check). Expect many minutes to hours for a fuzzed
# program. Recommend running in tmux / nohup and tailing the log.
#
# Usage (defaults = the u2k violation + config-fuzz-u2k-strong.yaml):
#   bash minimize_violation.sh
#   V=<violation-dir> CONFIG=<yaml> bash minimize_violation.sh      # e.g. the u2k violation + u2k-strong
#   INPUTS=100 SIMPLIFY=1 NOP=1 bash minimize_violation.sh          # deeper minimization
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
# Default target: the u2k violation (guest-user -> guest-kernel). It reproduces stronger than u2u,
# so minimize's per-step checks are more reliable. Inputs are LOADED from the violation dir (see
# RVZR_MINIMIZE_INPUT_DIR below), so the config's seed is irrelevant — we use the STRONG-sampling
# config for a more reliable per-step check. Override V=/CONFIG= for u2u.
V=${V:-$CAMP/out-fuzz-u2k/violation-260814-201415}
CONFIG=${CONFIG:-$CAMP/config-fuzz-u2k-strong.yaml}
OUT=${OUT:-$V/program.min.asm}
INDIR=${INDIR:-$V/min-inputs}
# use ALL the saved inputs (the counterexample is input #0 vs #50, so we need >=51; default = count of
# saved .bin, i.e. 100). minimize with too few generated inputs can NEVER form the [0,50] counterexample.
INPUTS=${INPUTS:-$(ls "$V"/input_*.bin 2>/dev/null | wc -l)}
ATTEMPTS=${ATTEMPTS:-3}
SIMPLIFY=${SIMPLIFY:-1}; NOP=${NOP:-0}; CONSTANT=${CONSTANT:-0}; MASK=${MASK:-0}; FENCE=${FENCE:-0}; COMMENT=${COMMENT:-1}

[ -f "$V/program.asm" ] || { echo "no program.asm in $V"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# NOTE: the Revizor doc regenerates inputs from minimize.yaml's data_generator_seed. That fails for a
# long fuzzing campaign (the input RNG advanced across ~19k TCs, so the per-TC seed doesn't cleanly
# regenerate this TC's inputs). So we instead LOAD the exact saved inputs via RVZR_MINIMIZE_INPUT_DIR
# (set just before the run) — the mechanism minimizer.py itself provides for this case. The config's
# seed is then irrelevant, so no seed-merge is needed; we use minimize.yaml straight.

FLAGS=""
[ "$SIMPLIFY" = 1 ] && FLAGS="$FLAGS --enable-simplification-pass true"
[ "$NOP" = 1 ]      && FLAGS="$FLAGS --enable-nop-pass true"
[ "$CONSTANT" = 1 ] && FLAGS="$FLAGS --enable-constant-pass true"
[ "$MASK" = 1 ]     && FLAGS="$FLAGS --enable-mask-pass true"
[ "$FENCE" = 1 ]    && FLAGS="$FLAGS --enable-fence-pass true"
[ "$COMMENT" = 1 ]  && FLAGS="$FLAGS --enable-comment-pass true"

# count "real" instructions (exclude labels ., macros ., comments, blanks)
count_instr () { grep -vE '^\s*(#|\.|$)' "$1" 2>/dev/null | grep -cvE '^\s*$'; }

echo "=================================================================="
echo " MINIMIZE violation"
echo "   violation : $V"
echo "   config    : $CONFIG"
echo "   module    : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "   inputs    : $INPUTS   attempts: $ATTEMPTS"
echo "   passes    : instruction${FLAGS:+,}$(echo "$FLAGS" | sed 's/--enable-//g; s/-pass true//g')"
echo "   output    : $OUT"
echo "   instrs before: $(count_instr "$V/program.asm")"
echo "=================================================================="
echo "[run] minimize starting ($(date +%H:%M:%S)) — this is SLOW; live log below:"
echo "------------------------------------------------------------------"
mkdir -p "$INDIR"
cd "$REPO"
# make minimize LOAD the exact saved inputs (incl. #50) instead of regenerating them from the seed —
# this is what makes PASS 1 reproduce at the same rate as the reproduce/replay scripts
export RVZR_MINIMIZE_INPUT_DIR="$V"
"$PY" revizor.py minimize -s "$BASE" -c "$CONFIG" -I "$V" \
    -t "$V/program.asm" -i "$INPUTS" -o "$OUT" --input-outdir "$INDIR" \
    --num-attempts "$ATTEMPTS" $FLAGS
RC=$?
echo "------------------------------------------------------------------"
if [ -f "$OUT" ]; then
    echo " minimized: $(count_instr "$OUT") instrs  (was $(count_instr "$V/program.asm"))   -> $OUT"
    echo " === surviving guest gadgets (the essence of the leak) ==="
    awk '/\.section \.data\.(vmvictim|vm)$/{p=1} /\.test_case_exit/{p=0} p' "$OUT" \
        | grep -vE '^\s*(#|$)' | sed 's/^/   /' | head -60
else
    echo " no minimized output produced — the violation likely did not reproduce under this config."
    echo " Try a larger INPUTS, confirm the module is the clean 1CD139EF build, and that this exact"
    echo " violation reproduces first (replay_fuzz.sh)."
fi
echo "=================================================================="
exit "$RC"
