#!/usr/bin/env bash
# Close the loop: is the propagation via the HIGH 64-bit product (rdx) into the
# L49 encoder `movsx rdx, byte [r14+rdx]` @0x17d8?
#
#   baseline : mul [r14+rsi]                       (rdx:rax = rax*V)          -> present
#   V1 imul  : mov r10,[r14+rsi] ; imul rax, r10   (writes rax low only; rdx UNTOUCHED)
#             -> dies  => secret travels via HIGH product rdx  => encoder = L49 (rdx-addressed)
#             -> lives => low product rax suffices => encoder downstream (rax-addressed: L77/L80)
#   V2 killrdx: baseline mul, then `xor rdx,rdx` right before the L49 load
#             -> dies  => L49 (address = rdx) IS an encoder (confirms)
#             -> lives => L49 not essential; look elsewhere
#
# Env: REPS=3  CLAUSE='cond, bpas'
# Usage: ./exp_chain.sh [violation-folder=violation-260701-234841]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:-violation-260701-234841}"; F="$VD/$name"
REPS="${REPS:-3}"; CLAUSE="${CLAUSE:-cond, bpas}"
cd "$RV" || exit 1
src="$F/minimized.asm"; [ -f "$src" ] || { echo "no minimized.asm in $F"; exit 1; }

ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
[ "$ssb" = Vulnerable ] || { echo "ERROR: SSBD not off (spec_store_bypass=$ssb)."; exit 1; }
if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then ins=( "$F"/min_inputs/input*.bin ); else ins=( "$F"/input*.bin ); fi
echo "inputs=${#ins[@]}  REPS=$REPS  CLAUSE=[$CLAUSE]"

# --- build variants ---
vbase="$F/min_chain_base.asm"; cp "$src" "$vbase"
# V1: mul[mem] -> mov r10,[mem]; imul rax, r10   (rdx untouched)
v1="$F/min_chain_imul.asm"
awk '/^[[:space:]]*mul qword ptr \[r14 \+ rsi\]/{ print "mov r10, qword ptr [r14 + rsi]"; print "imul rax, r10"; next } {print}' "$src" > "$v1"
# V2: insert `xor rdx, rdx` right before the L49 encoder load
v2="$F/min_chain_killrdx.asm"
awk '/^[[:space:]]*movsx rdx, byte ptr \[r14 \+ rdx\]/{ print "xor rdx, rdx" } {print}' "$src" > "$v2"

# sanity: substitutions happened
[ "$(grep -c 'imul rax, r10' "$v1")" -eq 1 ] || { echo "ERROR: V1 substitution failed"; exit 1; }
[ "$(grep -c 'xor rdx, rdx' "$v2")" -eq "$(( $(grep -c 'xor rdx, rdx' "$src") + 1 ))" ] || { echo "ERROR: V2 insertion failed"; exit 1; }
echo "V1 imul spliced; V2 killrdx inserted before L49."

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
run_asm() {  # $1=asm $2=tag
  local asm="$1" tag="$2" real=0 err=0 i log viol dur
  for i in $(seq 1 "$REPS"); do
    log="$F/exp5_${tag}_$i.log"
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

echo; echo "### baseline (mul [mem]) ###";           run_asm "$vbase" baseline; echo "  baseline: ${RES[baseline]}/$REPS"
[ "${RES[baseline]}" -ge 1 ] || { echo "baseline didn't reproduce; abort."; rm -f "$cfg"; exit 1; }
echo; echo "### V1 imul rax,r10 (rdx untouched) ###"; run_asm "$v1" imul;      echo "  imul: ${RES[imul]}/$REPS"
echo; echo "### V2 xor rdx before L49 ###";           run_asm "$v2" killrdx;   echo "  killrdx: ${RES[killrdx]}/$REPS"
rm -f "$cfg"

echo; echo "================ SUMMARY (present / $REPS) ================"
for t in baseline imul killrdx; do
  r="${RES[$t]:-?}"; e="${ERR[$t]:-0}"; flag=""; [ "$e" -gt 0 ] && flag="  !! INVALID"
  printf '  %-10s %s%s\n' "$t" "$r" "$flag"
done
echo "----------------------------------------------------------"
bi="${RES[imul]:-1}"; bk="${RES[killrdx]:-1}"
if [ "${ERR[imul]:-0}" -eq 0 ] && [ "$bi" -eq 0 ] && [ "$bk" -eq 0 ]; then
  echo "VERDICT: both die => propagation via HIGH product rdx => ENCODER = L49 (movsx rdx,[r14+rdx]). Chain confirmed."
elif [ "$bi" -ge 1 ]; then
  echo "VERDICT: imul survives => LOW product rax carries it => encoder is downstream rax-addressed (test L77/L80 next)."
else
  echo "VERDICT: mixed => inspect: imul=$bi killrdx=$bk (see logs)."
fi
echo "variants: $v1 $v2   logs: $F/exp5_*_*.log"
