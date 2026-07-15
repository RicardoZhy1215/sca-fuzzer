#!/usr/bin/env bash
# SOURCE: is the secret carried by the multiplicand rax (the store-forwarded
# value from L23->L31 @0x105a) or by V (the value loaded @0x219f)?
# rdx = high(rax * V) drives the L49 encoder; fix each operand in turn.
#
#   baseline : mov r10,[r14+rsi] ; mul r10
#   fix_rax  : mov rax, 0x7fffffff ; mov r10,[r14+rsi] ; mul r10   (rax constant, V normal)
#             -> dies  => rax (store-forward @0x105a) is the secret carrier => SOURCE pinned there
#             -> lives => rax irrelevant; the secret is V @0x219f itself
#   fix_V    : mov r10,[r14+rsi] ; mov r10, 0x7fffffff ; mul r10    (8B access happens, value replaced)
#             -> dies  => V's VALUE matters (re-confirms exp_hibits)
#             -> lives => only the access mattered (would contradict exp_hibits)
#
# Env: REPS=3  CLAUSE='cond, bpas'
# Usage: ./exp_source.sh [violation-folder=violation-260701-234841]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:-violation-260701-234841}"; F="$VD/$name"
REPS="${REPS:-3}"; CLAUSE="${CLAUSE:-cond, bpas}"; K=0x7fffffff
cd "$RV" || exit 1
src="$F/minimized.asm"; [ -f "$src" ] || { echo "no minimized.asm in $F"; exit 1; }

ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
[ "$ssb" = Vulnerable ] || { echo "ERROR: SSBD not off (spec_store_bypass=$ssb)."; exit 1; }
if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then ins=( "$F"/min_inputs/input*.bin ); else ins=( "$F"/input*.bin ); fi
echo "inputs=${#ins[@]}  REPS=$REPS  CLAUSE=[$CLAUSE]  const=$K"

vbase="$F/min_src_base.asm"; cp "$src" "$vbase"
vfa="$F/min_src_fixrax.asm"
awk -v K="$K" '/^[[:space:]]*mul qword ptr \[r14 \+ rsi\]/{ print "mov rax, " K; print "mov r10, qword ptr [r14 + rsi]"; print "mul r10"; next } {print}' "$src" > "$vfa"
vfv="$F/min_src_fixV.asm"
awk -v K="$K" '/^[[:space:]]*mul qword ptr \[r14 \+ rsi\]/{ print "mov r10, qword ptr [r14 + rsi]"; print "mov r10, " K; print "mul r10"; next } {print}' "$src" > "$vfv"
[ "$(grep -c 'mul r10' "$vfa")" -eq 1 ] && [ "$(grep -c 'mul r10' "$vfv")" -eq 1 ] || { echo "ERROR: substitution failed"; exit 1; }

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

declare -A RES ERR
run_asm() {
  local asm="$1" tag="$2" real=0 err=0 i log viol dur
  for i in $(seq 1 "$REPS"); do
    log="$F/exp6_${tag}_$i.log"
    ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
    if grep -aqiE 'operand type mismatch|Error appeared while assembling|Assembler messages|Traceback|Segmentation' "$log"; then err=$((err+1)); continue; fi
    viol=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
    dur=$(grep -aoE 'Duration: [0-9.]+' "$log" | grep -aoE '[0-9.]+' | tail -1)
    [ -n "$viol" ] && [ "$viol" -ge 1 ] 2>/dev/null && real=$((real+1))
    printf '    [%s %s/%s] Violations=%s (%ss)\n' "$tag" "$i" "$REPS" "${viol:-?}" "${dur:-?}"
  done
  RES["$tag"]=$real; ERR["$tag"]=$err
  [ "$err" -gt 0 ] && echo "    !! $tag: $err/$REPS FAILED TO ASSEMBLE/RUN (INVALID)"
}

echo; echo "### baseline ###";                run_asm "$vbase" baseline; echo "  baseline: ${RES[baseline]}/$REPS"
[ "${RES[baseline]}" -ge 1 ] || { echo "baseline didn't reproduce; abort."; rm -f "$cfg"; exit 1; }
echo; echo "### fix_rax (rax=$K, V normal) ###"; run_asm "$vfa" fix_rax; echo "  fix_rax: ${RES[fix_rax]}/$REPS"
echo; echo "### fix_V (V=$K, access kept) ###";  run_asm "$vfv" fix_V;   echo "  fix_V: ${RES[fix_V]}/$REPS"
rm -f "$cfg"

echo; echo "================ SUMMARY (present / $REPS) ================"
for t in baseline fix_rax fix_V; do
  r="${RES[$t]:-?}"; e="${ERR[$t]:-0}"; flag=""; [ "$e" -gt 0 ] && flag="  !! INVALID"
  printf '  %-9s %s%s\n' "$t" "$r" "$flag"
done
echo "----------------------------------------------------------"
fr="${RES[fix_rax]:-1}"; fv="${RES[fix_V]:-1}"
echo "fix_rax=$fr : 0 => rax (store-forward @0x105a) is the carrier;  >=1 => V @0x219f is the carrier."
echo "fix_V=$fv  : 0 => V's value matters (re-confirms exp_hibits);   >=1 => only the access mattered."
echo "variants: $vfa $vfv   logs: $F/exp6_*_*.log"
