#!/usr/bin/env bash
#
# minimize_040842.sh — shrink violation-260825-040842 (both-guests-KERNEL, ct-ni) to its leaking gadget.
#
# Inputs are LOADED from the violation dir (RVZR_MINIMIZE_INPUT_DIR) so the #39/#89 counterexample is
# present and the config seed is irrelevant. Config = the violation's own reproduce.yaml (both kernel,
# F+R, hpa_gpa, threshold 0.5). Passes: instruction-removal + simplification + comment.
#
# SLOW (O(#instrs x one violation check)); run in tmux/nohup.
#
# Usage:
#   bash minimize_040842.sh
#   NOP=1 bash minimize_040842.sh          # deeper (also the nop pass)
#   INPUTS=100 ATTEMPTS=3 bash minimize_040842.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$REPO/violation-260825-040842}
CONFIG=${CONFIG:-$V/reproduce.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/org-config.yaml
OUT=${OUT:-$V/program.min.asm}
INDIR=${INDIR:-$V/min-inputs}
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
echo " MINIMIZE  violation-260825-040842  (both guests KERNEL, ct-ni)"
echo "   violation : $V"
echo "   config    : $CONFIG   ($(grep -E 'analyser_stat_threshold' "$CONFIG" | awk '{print $2}') thr)"
echo "   module    : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)   ucode: $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)"
echo "   inputs    : $INPUTS (loaded)   attempts: $ATTEMPTS   counterexample: #39 vs #89"
echo "   passes    : instruction${FLAGS:+,}$(echo "$FLAGS" | sed 's/--enable-//g; s/-pass true//g')"
echo "   instrs before: $(count_instr "$V/program.asm")   ->  output: $OUT"
echo "=================================================================="
echo "[run] minimize starting ($(date +%H:%M:%S)) — SLOW; live log:"
echo "------------------------------------------------------------------"
mkdir -p "$INDIR"
cd "$REPO"
export RVZR_MINIMIZE_INPUT_DIR="$V"
"$PY" revizor.py minimize -s "$BASE" -c "$CONFIG" -I "$V" \
    -t "$V/program.asm" -i "$INPUTS" -o "$OUT" --input-outdir "$INDIR" \
    --num-attempts "$ATTEMPTS" $FLAGS
RC=$?
echo "------------------------------------------------------------------"
if [ -f "$OUT" ]; then
    echo " minimized: $(count_instr "$OUT") instrs  (was $(count_instr "$V/program.asm"))   -> $OUT"
    echo " === surviving guest gadgets ==="
    awk '/\.section \.data\.(vmvictim|vm)$/{p=1} /\.test_case_exit/{p=0} p' "$OUT" \
        | grep -vE '^\s*(#|$)' | sed 's/^/   /' | head -60
    echo ""
    echo " next: leaked values ->  bash $CAMP/leaked_values_040842.sh"
else
    echo " no minimized output — the violation likely did not reproduce under this config."
    echo " Confirm the module is clean 1CD139EF and that this violation reproduces first."
fi
echo "=================================================================="
exit "$RC"
