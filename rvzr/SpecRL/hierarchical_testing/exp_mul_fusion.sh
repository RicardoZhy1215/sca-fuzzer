#!/usr/bin/env bash
# Experiment 1: decouple "unaligned qword LOAD" from "memory-operand MUL path".
#
#   baseline:  mul qword ptr [r14 + rsi]              (fused load + multiply)
#   variant:   mov r10, qword ptr [r14 + rsi]         (plain unaligned 8B load)
#              mul r10                                 (register multiply)
#
# Semantically identical (rax = rax * mem; writes rdx:rax), same load address /
# same rsi mask -> the unaligned qword load is preserved; only the fusion is gone.
#
#   variant KILLS the leak  -> it's the memory-operand-MUL / fused load-op PATH.
#   variant KEEPS the leak  -> it's the unaligned qword LOAD itself (mul irrelevant).
#
# Env: REPS=5  CLAUSE='cond, bpas'
# Usage: ./exp_mul_fusion.sh [violation-folder=violation-260701-234841]
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

# build the variant: replace the single `mul qword ptr [r14 + rsi]` with mov+mul
var="$F/min_mulsplit.asm"
awk '/^[[:space:]]*mul qword ptr \[r14 \+ rsi\]/{
       print "mov r10, qword ptr [r14 + rsi]";
       print "mul r10";
       next } {print}' "$src" > "$var"
nsub=$(grep -c 'mul r10' "$var")
[ "$nsub" -eq 1 ] || { echo "ERROR: expected exactly 1 mul-substitution, got $nsub. Check the mul line."; exit 1; }
echo "variant written: $var  (mul [mem] -> mov r10,[mem]; mul r10)"
echo "--- variant diff ---"; diff <(grep -nE 'mul' "$src") <(grep -nE 'mul|mov r10' "$var") || true

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
    log="$F/exp1_${tag}_$i.log"
    ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
    viol=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
    dur=$(grep -aoE 'Duration: [0-9.]+' "$log" | grep -aoE '[0-9.]+' | tail -1)
    grep -aqiE 'Traceback|AssertionError|Error:|not.*instrument|failed' "$log" && err=$((err+1))
    if [ -n "$viol" ] && [ "$viol" -ge 1 ] 2>/dev/null; then real=$((real+1)); fi
    printf '    [%s %s/%s] Violations=%s (%ss)\n' "$tag" "$i" "$REPS" "${viol:-?}" "${dur:-?}"
  done
  [ "$err" -gt 0 ] && echo "    NOTE: $err/$REPS runs had an error/exception in the log (check $F/exp1_${tag}_*.log)"
  echo "$real"
}

echo; echo "### baseline: mul qword ptr [r14+rsi] ###"
base=$(run_asm "$src" baseline | tail -1)
echo "  baseline: present $base/$REPS"
[ "$base" -ge 1 ] || { echo "ERROR: baseline didn't reproduce; abort."; rm -f "$cfg"; exit 1; }

echo; echo "### variant: mov r10,[r14+rsi]; mul r10 ###"
vres=$(run_asm "$var" mulsplit | tail -1)
echo "  variant: present $vres/$REPS"
rm -f "$cfg"

echo; echo "================ SUMMARY (present / $REPS) ================"
printf '  %-20s %s\n' "baseline mul[mem]" "$base"
printf '  %-20s %s\n' "variant mov+mul" "$vres"
echo "----------------------------------------------------------"
if [ "$vres" -eq 0 ]; then
  echo "VERDICT: splitting the fused load KILLS it -> the memory-operand-MUL / fused load-op PATH is key."
elif [ "$vres" -ge 1 ]; then
  echo "VERDICT: leak survives with a plain load -> the UNALIGNED QWORD LOAD is key; MUL is incidental."
else
  echo "VERDICT: inconclusive."
fi
echo "logs: $F/exp1_*_*.log   variant: $var"
