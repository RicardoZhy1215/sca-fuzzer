#!/usr/bin/env bash
# Granularity sweep on the ONE geometry-critical access (the transmitter):
#   mul qword ptr [r14 + rsi]  @0x219f   (mask line = L44 in minimized.asm)
#
# pin_geometry.sh showed 8B-aligning this access (only this one) kills the leak.
# Here we sweep the mask from byte-granular down to cache-line-granular to find
# the EXACT address bit that carries the secret-dependent difference:
#   dies already at clear-bit0 (2B)  -> byte-level offset matters (sub-8B / partial-line STLF)
#   dies only at clear-bit2 (8B)     -> qword index matters (8B-granular transmitter)
#   dies only at clear-bit6 (64B)    -> merely cache-line-level (ordinary cache leak)
#
# Env: REPS=3  CLAUSE='cond, bpas'  MASK_LINE=44
# Usage: ./pin_mul.sh [violation-folder=violation-260701-234841]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:-violation-260701-234841}"; F="$VD/$name"
REPS="${REPS:-3}"; CLAUSE="${CLAUSE:-cond, bpas}"; ML="${MASK_LINE:-44}"
cd "$RV" || exit 1
src="$F/minimized.asm"; [ -f "$src" ] || { echo "no minimized.asm in $F"; exit 1; }

# masks: label -> 13-bit mask, from byte-granular to cache-line-granular
MASKS=(
  "byte_2B_clear0:0b1111111111111"
  "align2B_clear1:0b1111111111110"
  "align4B_clear2:0b1111111111100"
  "align8B_clear3:0b1111111111000"
  "align16B_clear4:0b1111111110000"
  "align32B_clear5:0b1111111100000"
  "align64B_clear6:0b1111111000000"
)

ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
[ "$ssb" = Vulnerable ] || { echo "ERROR: SSBD not off (spec_store_bypass=$ssb)."; exit 1; }
if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then ins=( "$F"/min_inputs/input*.bin ); isrc=min_inputs
else ins=( "$F"/input*.bin ); isrc=input; fi

line_txt=$(awk -v L="$ML" 'NR==L' "$src")
echo "mask line L$ML: $line_txt"
case "$line_txt" in *"and rsi"*"0b"*) : ;; *) echo "WARN: L$ML is not 'and rsi, 0b..' -> check MASK_LINE"; esac
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
    log="$F/pinmul_${tag}_$i.log"
    ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
    viol=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
    dur=$(grep -aoE 'Duration: [0-9.]+' "$log" | grep -aoE '[0-9.]+' | tail -1)
    if [ -n "$viol" ] && [ "$viol" -ge 1 ] 2>/dev/null; then real=$((real+1)); fi
    printf '    [%s %s/%s] Violations=%s (%ss)\n' "$tag" "$i" "$REPS" "${viol:-?}" "${dur:-?}"
  done
  echo "$real"
}

declare -A RES
for m in "${MASKS[@]}"; do
  lbl="${m%%:*}"; mask="${m#*:}"
  var="$F/min_mulmask_${lbl}.asm"
  awk -v L="$ML" -v M="$mask" 'NR==L{ sub(/0b[01]+/, M) } {print}' "$src" > "$var"
  echo; echo "### mul mask = $mask ($lbl) ###"
  RES["$lbl"]=$(run_asm "$var" "$lbl" | tail -1)
  echo "  $lbl: present ${RES[$lbl]}/$REPS"
done
rm -f "$cfg"

echo; echo "============ SUMMARY: mul @0x219f mask granularity (present / $REPS) ============"
for m in "${MASKS[@]}"; do
  lbl="${m%%:*}"; mask="${m#*:}"; r="${RES[$lbl]:-?}"
  mark=""; [ "$r" = 0 ] && mark="   <-- KILLED here"
  printf '  %-18s %s   %s%s\n' "$lbl" "$mask" "$r" "$mark"
done
echo "--------------------------------------------------------------------------------"
echo "The FIRST mask (top-down) that drops to 0 is the alignment threshold: the address"
echo "bit it clears is the one carrying the secret-dependent difference in the transmitter."
echo "variants: $F/min_mulmask_*.asm   logs: $F/pinmul_*_*.log"
