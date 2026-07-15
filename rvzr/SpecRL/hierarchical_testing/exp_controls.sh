#!/usr/bin/env bash
# Controls A & B: separate "8-byte ACCESS (microarch)" from "64-bit VALUE", and
# "single 8B load" from "64-bit value assembled from two 4B loads".
#
#   baseline : mov r10, qword [r14+rsi] ; mul r10                 (8B access, 64-bit value)  -> known present
#   ctrlA    : mov r10, qword [r14+rsi] ; and r10, 0xffffffff ; mul r10   (8B access, low-32 value)
#   ctrlB    : mov r10d,[r14+rsi] ; mov r11d,[r14+rsi+4] ; shl r11,32 ; or r10,r11 ; mul r10
#              (two 4B accesses assembled into the SAME 64-bit value; no single 8B load)
#
# Reading:
#   ctrlA present -> the 8B ACCESS itself is key; high 32 bits of the value don't matter
#   ctrlA absent  -> the 64-bit VALUE (high bits) propagates downstream, not the access width
#   ctrlB present -> it's 64-bit VALUE propagation, a single 8B load is NOT required
#   ctrlB absent  -> the single 8B UNALIGNED load's microarch (partial STLF) is required
#
# Env: REPS=5  CLAUSE='cond, bpas'
# Usage: ./exp_controls.sh [violation-folder=violation-260701-234841]
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

# replacement blocks (each replaces the single `mul qword ptr [r14 + rsi]` line)
blk_baseline=$'mov r10, qword ptr [r14 + rsi]\nmul r10'
blk_ctrlA=$'mov r10, qword ptr [r14 + rsi]\nand r10, 0xffffffff\nmul r10'
blk_ctrlB=$'mov r10d, dword ptr [r14 + rsi]\nmov r11d, dword ptr [r14 + rsi + 4]\nshl r11, 32\nor r10, r11\nmul r10'

gen_variant() {  # $1=replacement-block  $2=out
  local rf; rf="$(mktemp)"; printf '%s\n' "$1" > "$rf"
  awk -v RF="$rf" '/^[[:space:]]*mul qword ptr \[r14 \+ rsi\]/{ while((getline l < RF)>0) print l; close(RF); next } {print}' "$src" > "$2"
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

run_asm() {
  local asm="$1" tag="$2" real=0 err=0 i log viol dur
  for i in $(seq 1 "$REPS"); do
    log="$F/exp3_${tag}_$i.log"
    ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
    viol=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
    dur=$(grep -aoE 'Duration: [0-9.]+' "$log" | grep -aoE '[0-9.]+' | tail -1)
    grep -aqiE 'Traceback|AssertionError|Error:|not.*instrument|failed to|Segmentation' "$log" && err=$((err+1))
    if [ -n "$viol" ] && [ "$viol" -ge 1 ] 2>/dev/null; then real=$((real+1)); fi
    printf '    [%s %s/%s] Violations=%s (%ss)\n' "$tag" "$i" "$REPS" "${viol:-?}" "${dur:-?}"
  done
  [ "$err" -gt 0 ] && echo "    NOTE: $err/$REPS runs had an error/exception (check $F/exp3_${tag}_*.log)"
  echo "$real"
}

va="$F/min_ctrlA.asm"; vb="$F/min_ctrlB.asm"; vbase="$F/min_ctrlBASE.asm"
gen_variant "$blk_baseline" "$vbase"
gen_variant "$blk_ctrlA"    "$va"
gen_variant "$blk_ctrlB"    "$vb"
echo "--- ctrlB spliced region ---"; grep -nE 'r10|r11' "$vb" | sed -n '1,6p'

echo; echo "### baseline: 8B load, 64-bit value ###"
base=$(run_asm "$vbase" base | tail -1); echo "  baseline: present $base/$REPS"
[ "$base" -ge 1 ] || { echo "ERROR: baseline didn't reproduce; abort."; rm -f "$cfg"; exit 1; }
echo; echo "### Control A: 8B load, value masked to low 32 ###"
ra=$(run_asm "$va" ctrlA | tail -1); echo "  ctrlA: present $ra/$REPS"
echo; echo "### Control B: two 4B loads assembled to 64-bit ###"
rb=$(run_asm "$vb" ctrlB | tail -1); echo "  ctrlB: present $rb/$REPS"
rm -f "$cfg"

echo; echo "================ SUMMARY (present / $REPS) ================"
printf '  %-28s %s\n' "baseline 8B load / 64-bit val" "$base"
printf '  %-28s %s\n' "A: 8B load / low-32 value" "$ra"
printf '  %-28s %s\n' "B: two 4B loads / 64-bit val" "$rb"
echo "----------------------------------------------------------"
echo "A present => 8B ACCESS is key (high value bits irrelevant); A absent => 64-bit VALUE bits matter."
echo "B present => 64-bit VALUE propagation (single 8B load not required); B absent => single 8B unaligned load required."
echo "variants: $va $vb   logs: $F/exp3_*_*.log"
