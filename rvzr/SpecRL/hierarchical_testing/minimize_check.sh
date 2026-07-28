#!/usr/bin/env bash
# Minimize a violation to isolate the minimal leak gadget.
#
# Uses the folder's own minimize.yaml (ct + delayed-exception-handling, present:false
# faulty page, seed 2910). Inputs are REGENERATED from that seed (minimize mode takes
# a COUNT via -i, not files), so -i must match the original input count (100).
#
# Passes enabled (thorough one-shot, to strip down to the gadget + annotate it):
#   instruction  : remove instructions that aren't needed for the violation   [the key pass]
#   simplification: replace complex instrs with simpler ones
#   nop / constant / mask : blank operands / zero constants / shrink masks
#   label        : drop unused labels
#   comment      : annotate which mem accesses leak (load/store addr + L1D cache set)
#   input-diff   : drop inputs that don't contribute (finds the 2 violating inputs)
#
# No sudo needed. Runs MANY executor measurements, so run on a QUIET machine
# (same NMI/PMI caveat as reproduce) and expect several minutes.
#
# Usage:
#   bash minimize_check.sh                      # default folder, defaults below
#   NI=100 ATTEMPTS=5 TIMEOUT=1800 bash minimize_check.sh <violation-folder>
set -u
REPO=/home/hz25d/beyond_v1_v4/sca-fuzzer
PY=/home/hz25d/miniconda3/envs/specrl/bin/python
HT=rvzr/SpecRL/hierarchical_testing
NAME="${1:-violation-260721-212748}"
NI="${NI:-100}"                 # number of inputs (regenerated from seed); match original count
ATTEMPTS="${ATTEMPTS:-5}"       # more attempts -> tighter minimization, slower
TIMEOUT="${TIMEOUT:-none}"      # wall-clock cap (s); 'none'/0 = UNLIMITED (default)

V="$HT/$NAME"
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }
for f in "$V/minimize.yaml" "$V/program.asm" base.json; do
    [ -f "$f" ] || { echo "missing: $REPO/$f"; exit 1; }
done

if pgrep -af 'train\.py' >/dev/null 2>&1; then
    echo "WARNING: train.py is running -- it competes for the executor KM; results may be flaky."
fi
# SMT resets to ON after every reboot; with it on, measurements get NMI/PMI-corrupted
# and each one retries -> minimize crawls. Strongly recommended to disable it first.
if [ "$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)" = "on" ]; then
    echo "WARNING: SMT is ON -> measurements are slow/flaky. Disable it first:"
    echo "         echo off | sudo tee /sys/devices/system/cpu/smt/control"
fi
# Full minimization: input-memory passes (input-diff / input-seq) ON by default.
# They iterate the whole input region and are slow -- default TIMEOUT is unlimited.
# Set INPUT_PASS=0 to skip them for a quick program-only minimization.
INPUT_PASS="${INPUT_PASS:-1}"

OUT="$V/min.asm"
INDIR="$V/min-inputs"
mkdir -p "$INDIR"
log="$V/minimize.log"

echo "folder   : $NAME"
echo "config   : $V/minimize.yaml  (ct + delayed-exception-handling, present:false, seed 2910)"
echo "inputs   : $NI regenerated from seed"
echo "output   : program -> $OUT   inputs -> $INDIR"
echo "passes   : instruction, simplification, nop, constant, mask, label, comment; input-diff/seq=$INPUT_PASS"
echo "log      : $REPO/$log   (timeout ${TIMEOUT}s, num-attempts $ATTEMPTS)"
echo "SSBD(sys): $(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)"
echo "------------------------------------------------------------"

TO=(timeout --signal=INT "$TIMEOUT")
[ "$TIMEOUT" = "0" ] || [ "$TIMEOUT" = "none" ] && TO=()

"${TO[@]}" "$PY" ./revizor.py minimize \
    -s base.json \
    -c "$V/minimize.yaml" \
    -t "$V/program.asm" \
    -o "$OUT" \
    -i "$NI" \
    --input-outdir "$INDIR" \
    --num-attempts "$ATTEMPTS" \
    --enable-instruction-pass 1 \
    --enable-simplification-pass 1 \
    --enable-nop-pass 1 \
    --enable-constant-pass 1 \
    --enable-mask-pass 1 \
    --enable-label-pass 1 \
    --enable-comment-pass 1 \
    --enable-input-diff-pass "$INPUT_PASS" \
    --enable-input-seq-pass "$INPUT_PASS" \
    2>&1 | tee "$log"
rc=${PIPESTATUS[0]}

echo "------------------------------------------------------------"
echo "rc=$rc"
if [ -f "$OUT" ]; then
    echo "=== minimized program ($OUT) ==="
    grep -vE '^[[:space:]]*$' "$OUT"
else
    echo "no minimized program produced -- see $log (a mid-run trace EIO/NMI will abort it)"
fi
