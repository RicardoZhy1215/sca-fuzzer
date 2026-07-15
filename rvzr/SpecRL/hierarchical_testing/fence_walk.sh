#!/usr/bin/env bash
# LFENCE-walk: insert an LFENCE before EACH instruction, reproduce, and map where
# the leak dies (KILL) vs survives. The transition (last KILL -> first SURVIVE)
# brackets the DOWNSTREAM end of the speculative window = the real transmitter.
#
#   fence before i KILLS   => the leak's speculative access is at position >= i
#   fence before i SURVIVE => that access already happened before i
#   => speculative window (leak-relevant) = [store-bypass load ... last-KILL line]
#
# Env: REPS=2  CLAUSE='cond, bpas'  ASM=minimized.asm
# Usage: ./fence_walk.sh [violation-folder=violation-260701-234841]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:-violation-260701-234841}"; F="$VD/$name"
REPS="${REPS:-2}"; CLAUSE="${CLAUSE:-cond, bpas}"; ASM="${ASM:-minimized.asm}"
cd "$RV" || exit 1
src="$F/$ASM"; [ -f "$src" ] || { echo "ERROR: no $ASM in $F"; exit 1; }
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
[ "$ssb" = Vulnerable ] || { echo "ERROR: SSBD not off."; exit 1; }
[ -e /sys/rvzr_executor/enable_psfd ] && echo 0 > /sys/rvzr_executor/enable_psfd 2>/dev/null
# input selection: INDIR overrides; minimized2.asm needs its own min_inputs2
# (the minimizer's input-diff pass modifies the inputs, so each minimized gadget
#  is matched to its saved inputs; the original input*.bin only fits minimized.asm).
if [ -n "${INDIR:-}" ]; then ins=( "$F"/"$INDIR"/*input*.bin )
elif [ "$ASM" = "minimized2.asm" ]; then ins=( "$F"/min_inputs2/*input*.bin )
else ins=( "$F"/input*.bin ); fi

# instruction line numbers = lines that are NOT directives (.), comments (#), or blank
mapfile -t ILINES < <(awk 'NF && $0 !~ /^[[:space:]]*[#.]/ {print NR}' "$src")
echo "$ASM: ${#ILINES[@]} instruction lines  inputs=${#ins[@]}  REPS=$REPS  CLAUSE=[$CLAUSE]"
echo "(each fence variant = one reproduce run x REPS; est ~$(( ${#ILINES[@]} * REPS * 38 / 60 )) min)"

cfg="$(mktemp)"
grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^[[:space:]]*inputs_per_class:|^[[:space:]]*enable_priming:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$cfg"
grep -E '^(data_generator_seed|inputs_per_class):' "$F/reproduce.yaml" >> "$cfg"
{ echo "x86_executor_enable_ssbp_patch: false"; echo "contract_observation_clause: ct"
  echo "contract_execution_clause: [$CLAUSE]"; echo "enable_priming: true"; } >> "$cfg"

# baseline sanity
blog="$F/walk_baseline.log"
./revizor.py reproduce -s base.json -c "$cfg" -t "$src" -i "${ins[@]}" > "$blog" 2>&1
bv=$(grep -aoiE '^Violations: [0-9]+' "$blog" | grep -aoE '[0-9]+' | tail -1)
echo "baseline (no fence): Violations=${bv:-?}"
[ -n "$bv" ] && [ "$bv" -ge 1 ] 2>/dev/null || { echo "ABORT: baseline didn't reproduce."; rm -f "$cfg"; exit 1; }
echo "------------------------------------------------------------"
printf '%-6s %-40s %s\n' "line" "instruction" "present/$REPS"

declare -a MAP
for ln in "${ILINES[@]}"; do
  txt=$(awk -v L="$ln" 'NR==L{sub(/^[[:space:]]*/,""); sub(/[[:space:]]*#.*/,""); print}' "$src")
  var="$F/walk_$ln.asm"
  awk -v L="$ln" 'NR==L{print "lfence"} {print}' "$src" > "$var"
  real=0
  for r in $(seq 1 "$REPS"); do
    log="$F/walk_${ln}_$r.log"
    ./revizor.py reproduce -s base.json -c "$cfg" -t "$var" -i "${ins[@]}" > "$log" 2>&1
    grep -aqiE 'operand type mismatch|Error appeared while assembling|Traceback|Segmentation' "$log" && continue
    v=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
    [ -n "$v" ] && [ "$v" -ge 1 ] 2>/dev/null && real=$((real+1))
  done
  verdict=$([ "$real" -eq 0 ] && echo KILL || echo SURVIVE)
  MAP+=("$ln|$verdict|$txt")
  printf '%-6s %-40s %s  %s\n' "$ln" "${txt:0:40}" "$real" "$verdict"
  rm -f "$var"
done
rm -f "$cfg"

echo "------------------------------------------------------------"
# find last KILL followed by SURVIVE = downstream window boundary
prev_kill=""; boundary=""
for e in "${MAP[@]}"; do
  ln="${e%%|*}"; rest="${e#*|}"; vd="${rest%%|*}"
  if [ "$vd" = KILL ]; then prev_kill="$ln"
  elif [ "$vd" = SURVIVE ] && [ -n "$prev_kill" ] && [ -z "$boundary" ]; then boundary="$prev_kill"; fi
done
echo "SUMMARY:"
echo "  A fence KILLS through line $boundary, then SURVIVES after it."
echo "  => speculative window (leak-relevant) ends at line ~$boundary (the transmitter)."
echo "  => combined with SSBD (window opens at the store->load), the transient span is"
echo "     [store-bypass load ... line $boundary]."
echo "logs: $F/walk_*_*.log"
