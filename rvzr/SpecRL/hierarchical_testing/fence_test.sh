#!/usr/bin/env bash
# Confirm the speculation window by inserting LFENCE at chosen points in
# minimized.asm and reproducing. A fence that KILLS the leak closes the
# transient window at that point.
#
# Positions (original line numbers in minimized.asm; lfence inserted BEFORE each):
#   45  before_mul        -> right before the transmitter `mul [r14+rsi]` @0x219f
#   33  after_S1_pair     -> just after the @0x105a store->load bypass pair
#   25  after_S1_store    -> just after the @0x105a store (before its re-load)
#   62  before_S2_load    -> before the @0x1000 size-mismatch byte-load
#
# Reading:
#   before_mul KILLS      -> the mispeculated value reaches the `mul`; window ends there.
#   after_S1_* KILLS      -> the @0x105a store-bypass is the upstream source in the chain.
#
# Env: REPS=3  CLAUSE='cond, bpas'  POS="45:before_mul 33:after_S1_pair 25:after_S1_store 62:before_S2_load"
# Usage: ./fence_test.sh [violation-folder=violation-260701-234841]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:-violation-260701-234841}"; F="$VD/$name"
REPS="${REPS:-3}"; CLAUSE="${CLAUSE:-cond, bpas}"
POS="${POS:-45:before_mul 33:after_S1_pair 25:after_S1_store 62:before_S2_load}"
cd "$RV" || exit 1
src="$F/minimized.asm"; [ -f "$src" ] || { echo "no minimized.asm in $F"; exit 1; }

ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
[ "$ssb" = Vulnerable ] || { echo "ERROR: SSBD not off (spec_store_bypass=$ssb)."; exit 1; }
if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then ins=( "$F"/min_inputs/input*.bin ); isrc=min_inputs
else ins=( "$F"/input*.bin ); isrc=input; fi
echo "inputs=${#ins[@]} ($isrc)  REPS=$REPS  CLAUSE=[$CLAUSE]"

mkcfg() {
  local out; out="$(mktemp)"
  grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^[[:space:]]*inputs_per_class:|^[[:space:]]*enable_priming:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$out"
  grep -E '^(data_generator_seed|inputs_per_class):' "$F/reproduce.yaml" >> "$out"
  { echo "x86_executor_enable_ssbp_patch: false"
    echo "contract_observation_clause: ct"
    echo "contract_execution_clause: [$CLAUSE]"
    echo "enable_priming: true"; } >> "$out"
  echo "$out"
}
cfg="$(mkcfg)"

run_asm() {
  local asm="$1" tag="$2" real=0 i log viol dur
  for i in $(seq 1 "$REPS"); do
    log="$F/fence_${tag}_$i.log"
    ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
    viol=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
    dur=$(grep -aoE 'Duration: [0-9.]+' "$log" | grep -aoE '[0-9.]+' | tail -1)
    if [ -n "$viol" ] && [ "$viol" -ge 1 ] 2>/dev/null; then real=$((real+1)); fi
    printf '    [%s %s/%s] Violations=%s (%ss)\n' "$tag" "$i" "$REPS" "${viol:-?}" "${dur:-?}"
  done
  echo "$real"
}

echo "### baseline (no fence) ###"
base=$(run_asm "$src" baseline | tail -1)
echo "  baseline: present $base/$REPS"
[ "$base" -ge 1 ] || { echo "ERROR: baseline didn't reproduce; abort."; rm -f "$cfg"; exit 1; }

declare -A RES
for p in $POS; do
  ln="${p%%:*}"; lbl="${p#*:}"
  var="$F/min_fence_${lbl}.asm"
  awk -v L="$ln" 'NR==L{ print "lfence" } {print}' "$src" > "$var"
  echo; echo "### lfence before L$ln ($lbl) ###"
  RES["$lbl"]=$(run_asm "$var" "$lbl" | tail -1)
  echo "  $lbl: present ${RES[$lbl]}/$REPS"
done
rm -f "$cfg"

echo; echo "================ SUMMARY: LFENCE positions (present / $REPS) ================"
printf '  %-18s %s   (no fence)\n' "baseline" "$base"
for p in $POS; do
  lbl="${p#*:}"; r="${RES[$lbl]:-?}"
  mark=""; [ "$r" = 0 ] && mark="   <-- fence KILLS it here"
  printf '  %-18s %s%s\n' "$lbl" "$r" "$mark"
done
echo "----------------------------------------------------------------------------"
echo "before_mul at 0 confirms the transient window feeds the mul; an after_S1_* at 0"
echo "implicates the @0x105a store-bypass as the upstream source."
echo "variants: $F/min_fence_*.asm   logs: $F/fence_*_*.log"
