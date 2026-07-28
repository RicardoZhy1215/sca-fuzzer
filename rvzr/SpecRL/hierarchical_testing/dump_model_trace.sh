#!/usr/bin/env bash
# Dump the MODEL's per-instruction contract trace for min.asm, to see exactly what the
# CT+DEH speculator observes in the fault shadow (does it execute/observe the leaking
# store, or skip it?). This pinpoints the contract gap without guessing.
#
# It enables dbg_model + ctrace dump logging, runs reproduce on min.asm with the
# minimized inputs, and saves the full model trace. Compare the contract trace of the
# two violating inputs: identical ctrace (they share a hash) but the hardware differs
# => whatever data-dependent access the model OMITS is the gap.
#
# No sudo. Usage: bash dump_model_trace.sh   [violation-folder]
set -u
REPO=/home/hz25d/beyond_v1_v4/sca-fuzzer
PY=/home/hz25d/miniconda3/envs/specrl/bin/python
HT=rvzr/SpecRL/hierarchical_testing
NAME="${1:-violation-260721-212748}"
V="$HT/$NAME"

cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }
for f in "$V/reproduce.yaml" "$V/min.asm" base.json; do
    [ -f "$f" ] || { echo "missing: $REPO/$f"; exit 1; }
done
shopt -s nullglob
inputs=( "$V/min-inputs"/min_input*.bin )
shopt -u nullglob
[ ${#inputs[@]} -gt 0 ] || { echo "no min-inputs/min_input*.bin"; exit 1; }

# config with model-debug + contract-trace dump logging enabled
cfg="$V/reproduce_dbgmodel.yaml"
grep -vE '^[[:space:]]*logging_modes:|^[[:space:]]*-[[:space:]]*(info|stat|dbg_)' "$V/reproduce.yaml" > "$cfg"
cat >> "$cfg" <<'YAML'
logging_modes:
  - dbg_model
  - dbg_dump_ctraces
  - dbg_dump_htraces
  - dbg_violation
YAML

log="$V/model_trace.log"
echo "program : $V/min.asm"
echo "inputs  : ${#inputs[@]} (min-inputs)"
echo "config  : $cfg  (dbg_model + ctrace/htrace dump)"
echo "log     : $REPO/$log"
echo "------------------------------------------------------------"
"$PY" ./revizor.py reproduce -s base.json -c "$cfg" -t "$V/min.asm" -i "${inputs[@]}" > "$log" 2>&1
rc=$?
echo "rc=$rc   (log written)"
echo
echo "=== detailed contract traces (per counterexample input) ==="
grep -aA1 -iE "Contract trace \(detailed\)|Contract trace \(hash\)" "$log" | head -40
echo
echo "=== model instruction trace around the fault (first 60 dbg_model lines) ==="
grep -aiE "dbg_model|speculat|fault|skip|\bpc:|\bmem:|RIP" "$log" | head -60
echo
echo "Full trace in: $REPO/$log  (send it over for analysis)"
