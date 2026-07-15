#!/usr/bin/env bash
# Experiment 2: load-WIDTH sweep at the same (odd) address. Fixes the register
# multiply, varies only the transmitter load width, to separate:
#   - "8-byte SPAN partially overlaps a store"  (partial-overlap STLF)   vs
#   - "the ADDRESS bit itself matters"          (partial address match, SPOILER-like)
#
#   8B  mov   r10,  qword ptr [r14 + rsi] ; mul r10
#   4B  mov   r10d, dword ptr [r14 + rsi] ; mul r10
#   2B  movzx r10,  word  ptr [r14 + rsi] ; mul r10
#   1B  movzx r10,  byte  ptr [r14 + rsi] ; mul r10
#
# Reading:
#   only 8B leaks (4/2/1B die)  -> the 8-byte span / partial overlap is essential  (partial-overlap STLF)
#   narrow loads also leak      -> width-independent -> the ADDRESS (bit0) matters  (partial addr match / SPOILER-like)
#
# Env: REPS=5  CLAUSE='cond, bpas'
# Usage: ./exp_load_width.sh [violation-folder=violation-260701-234841]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:-violation-260701-234841}"; F="$VD/$name"
REPS="${REPS:-5}"; CLAUSE="${CLAUSE:-cond, bpas}"
cd "$RV" || exit 1
src="$F/minimized.asm"; [ -f "$src" ] || { echo "no minimized.asm in $F"; exit 1; }

ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
[ "$ssb" = Vulnerable ] || { echo "ERROR: SSBD not off (spec_store_bypass=$ssb)."; exit 1; }
if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then ins=( "$F"/min_inputs/input*.bin ); isrc=min_inputs
else ins=( "$F"/input*.bin ); isrc=input; fi
echo "inputs=${#ins[@]} ($isrc)  REPS=$REPS  CLAUSE=[$CLAUSE]"

# width label | load instruction (mul r10 appended for all)
WIDTHS=(
  "8B|mov r10, qword ptr [r14 + rsi]"
  "4B|mov r10d, dword ptr [r14 + rsi]"
  "2B|movzx r10, word ptr [r14 + rsi]"
  "1B|movzx r10, byte ptr [r14 + rsi]"
)

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
  local asm="$1" tag="$2" real=0 err=0 i log viol dur
  for i in $(seq 1 "$REPS"); do
    log="$F/exp2_${tag}_$i.log"
    ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
    viol=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
    dur=$(grep -aoE 'Duration: [0-9.]+' "$log" | grep -aoE '[0-9.]+' | tail -1)
    grep -aqiE 'Traceback|AssertionError|Error:|not.*instrument|failed to' "$log" && err=$((err+1))
    if [ -n "$viol" ] && [ "$viol" -ge 1 ] 2>/dev/null; then real=$((real+1)); fi
    printf '    [%s %s/%s] Violations=%s (%ss)\n' "$tag" "$i" "$REPS" "${viol:-?}" "${dur:-?}"
  done
  [ "$err" -gt 0 ] && echo "    NOTE: $err/$REPS runs had an error/exception (check $F/exp2_${tag}_*.log)"
  echo "$real"
}

declare -A RES
for w in "${WIDTHS[@]}"; do
  lbl="${w%%|*}"; ld="${w#*|}"
  var="$F/min_ldw_${lbl}.asm"
  awk -v LD="$ld" '/^[[:space:]]*mul qword ptr \[r14 \+ rsi\]/{ print LD; print "mul r10"; next } {print}' "$src" > "$var"
  # verify substitution happened exactly once
  if [ "$(grep -c 'mul r10' "$var")" -ne 1 ]; then echo "ERROR: substitution failed for $lbl"; rm -f "$cfg"; exit 1; fi
  echo; echo "### $lbl load: $ld ; mul r10 ###"
  RES["$lbl"]=$(run_asm "$var" "$lbl" | tail -1)
  echo "  $lbl: present ${RES[$lbl]}/$REPS"
done
rm -f "$cfg"

echo; echo "================ SUMMARY: transmitter load width (present / $REPS) ================"
for w in "${WIDTHS[@]}"; do
  lbl="${w%%|*}"; r="${RES[$lbl]:-?}"
  mark=""; [ "$r" = 0 ] && mark="   <-- dies at this width"
  printf '  %-4s %s%s\n' "$lbl" "$r" "$mark"
done
echo "--------------------------------------------------------------------------------"
b8="${RES[8B]:-0}"; b1="${RES[1B]:-0}"; b2="${RES[2B]:-0}"; b4="${RES[4B]:-0}"
if [ "$b8" -ge 1 ] && [ "$b4" -eq 0 ] && [ "$b2" -eq 0 ] && [ "$b1" -eq 0 ]; then
  echo "VERDICT: only 8B leaks -> the 8-byte SPAN / partial overlap is essential = partial-overlap STLF."
elif [ "$b1" -ge 1 ]; then
  echo "VERDICT: even a 1B load at the odd address leaks -> width-INDEPENDENT = the ADDRESS/bit0 drives it"
  echo "         (partial address match / SPOILER-like), not span overlap."
else
  echo "VERDICT: mixed -> threshold at the smallest width that still leaks (see table); interpret from there."
fi
echo "variants: $F/min_ldw_*.asm   logs: $F/exp2_*_*.log"
