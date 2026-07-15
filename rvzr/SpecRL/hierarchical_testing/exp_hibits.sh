#!/usr/bin/env bash
# Value-mask sweep (CORRECTED): localize which HIGH bits of the loaded 64-bit
# value matter. Masks the loaded value r10 before `mul r10`, via
#   movabs r11, MASK ; and r10, r11
# (x86-64 `and r64, imm` only takes imm32 — a 64-bit mask MUST go through a reg;
#  the earlier `and r10, 0xffffffff` was an assembler error, not a real control.)
#
# Uniform structure for every variant (baseline uses an all-ones no-op mask, so
# instruction count is held constant):
#   mov r10, qword ptr [r14 + rsi]
#   movabs r11, MASK
#   and r10, r11
#   mul r10
#
# Phase 1 (cumulative keep-low): find the threshold byte that flips absent->present.
# Phase 2 (low32 + one high byte): isolate which high byte restores it.
#
# Env: REPS=3  CLAUSE='cond, bpas'
# Usage: ./exp_hibits.sh [violation-folder=violation-260701-234841]
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

# label | 64-bit mask (kept bits)
CFG=(
  "full_noop|0xffffffffffffffff"     # control: expect present (== baseline)
  "keeplow32|0x00000000ffffffff"     # redone Control A: low 32 only
  "keeplow40|0x000000ffffffffff"     # + byte 32-39
  "keeplow48|0x0000ffffffffffff"     # + byte 40-47
  "keeplow56|0x00ffffffffffffff"     # + byte 48-55
  "lo32+b40_47|0x0000ff00ffffffff"   # low32 plus only byte 40-47
  "lo32+b48_55|0x00ff0000ffffffff"   # low32 plus only byte 48-55
  "lo32+b56_63|0xff000000ffffffff"   # low32 plus only byte 56-63
)

gen_variant() {  # $1=mask $2=out
  local rf; rf="$(mktemp)"
  printf 'mov r10, qword ptr [r14 + rsi]\nmovabs r11, %s\nand r10, r11\nmul r10\n' "$1" > "$rf"
  awk -v RF="$rf" '/^[[:space:]]*mul qword ptr \[r14 \+ rsi\]/{ while((getline l<RF)>0) print l; close(RF); next } {print}' "$src" > "$2"
  rm -f "$rf"
}

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
run_asm() {  # $1=asm $2=tag ; sets RES[tag], ERR[tag]
  local asm="$1" tag="$2" real=0 err=0 i log viol dur
  for i in $(seq 1 "$REPS"); do
    log="$F/exp4_${tag}_$i.log"
    ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
    if grep -aqiE 'operand type mismatch|Error appeared while assembling|Assembler messages|Traceback|Segmentation' "$log"; then err=$((err+1)); continue; fi
    viol=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
    dur=$(grep -aoE 'Duration: [0-9.]+' "$log" | grep -aoE '[0-9.]+' | tail -1)
    if [ -n "$viol" ] && [ "$viol" -ge 1 ] 2>/dev/null; then real=$((real+1)); fi
    printf '    [%s %s/%s] Violations=%s (%ss)\n' "$tag" "$i" "$REPS" "${viol:-?}" "${dur:-?}"
  done
  RES["$tag"]=$real; ERR["$tag"]=$err
  [ "$err" -gt 0 ] && echo "    !! $tag: $err/$REPS runs FAILED TO ASSEMBLE/RUN (result INVALID)"
}

for c in "${CFG[@]}"; do
  lbl="${c%%|*}"; mask="${c#*|}"
  var="$F/min_hib_${lbl//[+]/_}.asm"
  gen_variant "$mask" "$var"
  echo; echo "### $lbl  keep=$mask ###"
  run_asm "$var" "$lbl"
  echo "  $lbl: present ${RES[$lbl]}/$REPS${ERR[$lbl]:+  (errors: ${ERR[$lbl]})}"
done
rm -f "$cfg"

echo; echo "================ SUMMARY: value high-bit localization (present / $REPS) ================"
for c in "${CFG[@]}"; do
  lbl="${c%%|*}"; mask="${c#*|}"; r="${RES[$lbl]:-?}"; e="${ERR[$lbl]:-0}"
  flag=""; [ "$e" -gt 0 ] && flag="   !! INVALID ($e build/run errors)"
  printf '  %-12s %-20s %s%s\n' "$lbl" "$mask" "$r" "$flag"
done
echo "----------------------------------------------------------------------------------------"
echo "Sanity: full_noop must be present. Phase1: first keeplowNN (32->40->48->56) that turns"
echo "present pins the threshold byte. Phase2: whichever lo32+bXX is present isolates that byte."
echo "variants: $F/min_hib_*.asm   logs: $F/exp4_*_*.log"
