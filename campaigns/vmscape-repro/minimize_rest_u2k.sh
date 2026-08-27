#!/usr/bin/env bash
#
# minimize_rest_u2k.sh — minimize the remaining refuzz-u2k violations, ONE AT A TIME, in order.
#
# STRICTLY SEQUENTIAL: a plain for-loop; each `minimize` call blocks until it fully finishes before
# the next one starts. Never runs two against the executor at once (that wedges it).
#
# By default it processes every out-refuzz-u2k/violation-* that does NOT yet have program.min.asm
# (so 090802, already minimized, is skipped) — i.e. 111402, 121434, 125257, oldest-... in a fixed order.
#
# Each one: instruction pass (remove) + simplification + comment, inputs LOADED from the violation dir
# (RVZR_MINIMIZE_INPUT_DIR), config = config-refuzz-u2k.yaml (threshold 0.5). Output -> program.min.asm.
#
# Usage:
#   bash minimize_rest_u2k.sh                 # all pending, in order
#   FORCE=1 bash minimize_rest_u2k.sh         # re-minimize even those already done
#   ONLY="violation-260817-111402 violation-260817-121434" bash minimize_rest_u2k.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
OUTDIR=$CAMP/out-refuzz-u2k
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
ATTEMPTS=${ATTEMPTS:-3}
FORCE=${FORCE:-0}

[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
count_instr () { grep -vE '^\s*(#|\.|$)' "$1" 2>/dev/null | grep -cvE '^\s*$'; }

# build the ordered work list (oldest-first for determinism), skipping already-minimized unless FORCE
if [ -n "${ONLY:-}" ]; then
    WORK=(); for n in $ONLY; do WORK+=("$OUTDIR/$n"); done
else
    WORK=()
    for V in $(ls -d "$OUTDIR"/violation-* 2>/dev/null | sort); do
        [ -f "$V/program.asm" ] || continue
        if [ "$FORCE" != "1" ] && [ -f "$V/program.min.asm" ]; then continue; fi
        WORK+=("$V")
    done
fi
[ "${#WORK[@]}" -gt 0 ] || { echo "nothing to minimize (all done? use FORCE=1). Dir: $OUTDIR"; exit 0; }

echo "=================================================================="
echo " SEQUENTIAL MINIMIZE — ${#WORK[@]} violation(s), one at a time, in order"
echo "   module : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
echo "   config : $CONFIG   attempts: $ATTEMPTS"
for V in "${WORK[@]}"; do echo "     - $(basename "$V")  ($(count_instr "$V/program.asm") instrs)"; done
echo "=================================================================="
cd "$REPO"

i=0; FAIL=0
for V in "${WORK[@]}"; do
    i=$((i+1)); name=$(basename "$V")
    cfg=$CONFIG; [ -f "$cfg" ] || cfg=$V/reproduce.yaml
    inputs=$(ls "$V"/input_*.bin 2>/dev/null | wc -l)
    echo ""
    echo ">>>>>>>>>>>>>>>> [$i/${#WORK[@]}] $name  ($(date +%H:%M:%S)) >>>>>>>>>>>>>>>>"
    echo "   $(count_instr "$V/program.asm") instrs, $inputs inputs (loaded), config $(basename "$cfg")"

    export RVZR_MINIMIZE_INPUT_DIR="$V"
    # foreground (no '&'): the shell blocks here until this minimize process fully exits;
    # only then does rc get set and the loop advance to the next violation.
    "$PY" revizor.py minimize -s "$BASE" -c "$cfg" -I "$V" \
        -t "$V/program.asm" -i "$inputs" -o "$V/program.min.asm" --input-outdir "$V/min-inputs" \
        --num-attempts "$ATTEMPTS" \
        --enable-simplification-pass true --enable-comment-pass true
    rc=$?
    if [ -f "$V/program.min.asm" ] && [ "$(count_instr "$V/program.min.asm")" -gt 0 ]; then
        echo "   => $name minimized: $(count_instr "$V/program.min.asm") instrs (was $(count_instr "$V/program.asm"))  [rc=$rc]"
    else
        echo "   => $name produced NO minimized output (violation didn't reproduce under minimize) [rc=$rc]"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "=================================================================="
echo " DONE — processed ${#WORK[@]}, no-output: $FAIL"
for V in "${WORK[@]}"; do
    m=$([ -f "$V/program.min.asm" ] && count_instr "$V/program.min.asm" || echo "-")
    echo "   $(basename "$V") : $(count_instr "$V/program.asm") -> $m instrs"
done
echo "=================================================================="