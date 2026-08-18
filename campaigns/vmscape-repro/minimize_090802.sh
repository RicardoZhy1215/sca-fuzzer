#!/usr/bin/env bash
#
# minimize_090802.sh — shrink the refuzz-u2k violation-260817-090802 to its essential leaking gadget.
#
# This is a GOOD minimize target: it reproduces 10/10 at threshold 0.5, so minimize's per-step
# violation checks are reliable (no drift to fight). Counterexample = input #34 vs #84 (both in 0-99).
#
# The saved inputs are LOADED (RVZR_MINIMIZE_INPUT_DIR) rather than regenerated from the seed, so the
# [#34,#84] counterexample is always present and the config's seed is irrelevant. Config =
# config-refuzz-u2k.yaml (ct-ni, F+R, hpa_gpa, threshold 0.5, inputs_per_class 2) — the campaign config.
#
# Passes: instruction (remove instrs, ON by default) + simplification + comment. Add NOP=1 for a
# deeper pass. SLOW: O(#instructions x one full violation check) — run in tmux/nohup and tail it.
#
# Usage:
#   bash minimize_090802.sh
#   NOP=1 bash minimize_090802.sh          # also try the nop pass (deeper, slower)
#   INPUTS=100 ATTEMPTS=3 bash minimize_090802.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/reproduce.yaml          # fallback if the campaign config got removed
OUT=${OUT:-$V/program.min.asm}
INDIR=${INDIR:-$V/min-inputs}
# use ALL saved inputs so the [#34,#84] counterexample is present (need > 84). Default = count = 100.
INPUTS=${INPUTS:-$(ls "$V"/input_*.bin 2>/dev/null | wc -l)}
ATTEMPTS=${ATTEMPTS:-3}
SIMPLIFY=${SIMPLIFY:-1}; NOP=${NOP:-0}; CONSTANT=${CONSTANT:-0}; MASK=${MASK:-0}; FENCE=${FENCE:-0}; COMMENT=${COMMENT:-1}

[ -f "$V/program.asm" ] || { echo "no program.asm in $V"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

FLAGS=""
[ "$SIMPLIFY" = 1 ] && FLAGS="$FLAGS --enable-simplification-pass true"
[ "$NOP" = 1 ]      && FLAGS="$FLAGS --enable-nop-pass true"
[ "$CONSTANT" = 1 ] && FLAGS="$FLAGS --enable-constant-pass true"
[ "$MASK" = 1 ]     && FLAGS="$FLAGS --enable-mask-pass true"
[ "$FENCE" = 1 ]    && FLAGS="$FLAGS --enable-fence-pass true"
[ "$COMMENT" = 1 ]  && FLAGS="$FLAGS --enable-comment-pass true"

count_instr () { grep -vE '^\s*(#|\.|$)' "$1" 2>/dev/null | grep -cvE '^\s*$'; }

echo "=================================================================="
echo " MINIMIZE  violation-260817-090802  (refuzz-u2k, reproduces 10/10)"
echo "   violation : $V"
echo "   config    : $CONFIG   ($(grep -E 'analyser_stat_threshold' "$CONFIG" | awk '{print $2}') thr)"
echo "   module    : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)   (want clean 1CD139EF)"
echo "   inputs    : $INPUTS (loaded)   attempts: $ATTEMPTS   counterexample: #34 vs #84"
echo "   passes    : instruction${FLAGS:+,}$(echo "$FLAGS" | sed 's/--enable-//g; s/-pass true//g')"
echo "   instrs before: $(count_instr "$V/program.asm")   ->  output: $OUT"
echo "=================================================================="
echo "[run] minimize starting ($(date +%H:%M:%S)) — SLOW; live log below:"
echo "------------------------------------------------------------------"
mkdir -p "$INDIR"
cd "$REPO"
# LOAD the exact saved inputs (incl. #84) instead of regenerating from the seed — this is what makes
# PASS 1 reproduce at the same 10/10 rate as replay_all.sh.
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
    echo ""
    echo " next: verify it still triggers ->"
    echo "   N=10 MIN=$OUT V=$V CONFIG=$CONFIG bash $CAMP/verify_min.sh"
else
    echo " no minimized output — PASS 1 likely didn't reproduce. Confirm the module is clean 1CD139EF,"
    echo " that this violation still reproduces (bash replay_refuzz_u2k.sh), and perf_env.sh is applied."
fi
echo "=================================================================="
exit "$RC"
