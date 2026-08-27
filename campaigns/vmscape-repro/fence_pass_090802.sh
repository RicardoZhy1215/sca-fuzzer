#!/usr/bin/env bash
#
# fence_pass_090802.sh — root-cause doc "Insert Speculation Fences".
#
# Runs the minimizer's fence pass on program.min.asm: it inserts an LFENCE after every instruction
# and keeps each fence only if the violation STILL occurs with it. In the result (program.min.fenced.asm)
# the instructions left WITHOUT a trailing lfence are the minimal speculative window — the region that
# must run speculatively to leak. Everything fenced is just data setup.
#
# Inputs are LOADED (RVZR_MINIMIZE_INPUT_DIR) so the #34/#84 counterexample is present; instruction
# pass OFF (program unchanged, only fences added). SLOW-ish: ~one reproduce per instruction x attempts.
#
# Usage:
#   bash fence_pass_090802.sh
#   ATTEMPTS=5 bash fence_pass_090802.sh      # stricter per-fence check (weak/drifty signal)
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/reproduce.yaml
PROG=${PROG:-$V/program.min.asm}
OUT=${OUT:-$V/program.min.fenced.asm}
INDIR=${INDIR:-$V/fence-inputs}
INPUTS=${INPUTS:-$(ls "$V"/input_*.bin 2>/dev/null | wc -l)}
ATTEMPTS=${ATTEMPTS:-3}

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

echo "=================================================================="
echo " FENCE PASS — 090802: localize the speculative window"
echo "   program : $PROG   ($(grep -vE '^\s*(#|\.|$)' "$PROG" | grep -cvE '^\s*$') instrs)"
echo "   config  : $CONFIG   inputs: $INPUTS (loaded)   attempts: $ATTEMPTS"
echo "   output  : $OUT"
echo "   module  : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "=================================================================="
echo "[run] fence pass starting ($(date +%H:%M:%S)) — one reproduce per instruction; live log:"
echo "------------------------------------------------------------------"
mkdir -p "$INDIR"
export RVZR_MINIMIZE_INPUT_DIR="$V"
cd "$REPO"
"$PY" revizor.py minimize -s "$BASE" -c "$CONFIG" -I "$V" \
    -t "$PROG" -i "$INPUTS" -o "$OUT" --input-outdir "$INDIR" --num-attempts "$ATTEMPTS" \
    --enable-instruction-pass false \
    --enable-fence-pass true \
    --enable-comment-pass true
RC=$?
echo "------------------------------------------------------------------"
if [ ! -f "$OUT" ]; then
    echo " no fenced output produced — the violation likely didn't reproduce under the fence pass."
    echo " Confirm the signal first (bash $CAMP/verify_min_090802.sh) and retry with ATTEMPTS=5."
    exit "$RC"
fi

echo " fenced program: $OUT"
echo " lfences inserted: $(grep -c '^\s*lfence' "$OUT")"
echo ""
echo " === SPECULATIVE WINDOW: instructions NOT followed by an lfence (the leak region) ==="
# an instruction is 'un-fenced' if the next real code line is NOT lfence. Walk the file, remember the
# previous instruction, and when we DON'T see an lfence before the next instruction, flag the previous.
awk '
  function isinstr(l){ return (l ~ /^[[:space:]]*[a-z]/) && (l !~ /^[[:space:]]*(lfence|nop|\.)/) && (l !~ /^[[:space:]]*#/) }
  /^[[:space:]]*lfence/ { fenced[previnstr]=1; next }
  /# mem access:/ { memacc[previnstr]=$0; next }
  isinstr($0) {
    if (previnstr != "" && !(previnstr in fenced)) {
      printf "   >>> %s%s\n", prevtext, (previnstr in memacc ? "   " memacc[previnstr] : "")
    }
    previnstr=NR; prevtext=$0
  }
  END {
    if (previnstr != "" && !(previnstr in fenced))
      printf "   >>> %s%s\n", prevtext, (previnstr in memacc ? "   " memacc[previnstr] : "")
  }
' "$OUT"
echo ""
echo " (full fenced program in $OUT; lines above have NO trailing lfence => must run speculatively.)"
echo "=================================================================="
exit "$RC"
