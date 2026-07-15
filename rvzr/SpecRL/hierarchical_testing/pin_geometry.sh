#!/usr/bin/env bash
# Pin the geometry of the beyond-bpas SSB violation by a TARGETED single-access
# alignment sweep: for each memory-access whose instrumentation mask leaves it
# sub-8B-aligned, tighten ONLY that one mask to 8B alignment (0b1111111111000)
# and reproduce. The access(es) whose alignment KILLS the violation are the
# ones whose exact offset / partial-overlap geometry drives the leak.
#
# Unlike whole-program de-align, this changes one address at a time, so it
# localizes WHICH store->load pair matters (Site-1 @0x105a vs Site-2 @0x1000).
#
# Baseline (unmodified) should reproduce; a variant that drops to 0 is decisive.
#
# Env: REPS=3  CLAUSE='cond, bpas'
# Usage: ./pin_geometry.sh [violation-folder=violation-260701-234841]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:-violation-260701-234841}"; F="$VD/$name"
REPS="${REPS:-3}"; CLAUSE="${CLAUSE:-cond, bpas}"
ALIGNED='0b1111111111000'   # clear low 3 bits -> 8-byte aligned
cd "$RV" || exit 1
src="$F/minimized.asm"; [ -f "$src" ] || { echo "no minimized.asm in $F"; exit 1; }

# candidate mask lines (line# in minimized.asm : label). Only sub-8B-aligned
# masks that feed a memory access are worth tightening.
CANDS=(
  "22:S1_store_0x105a"
  "28:S1_load_0x105a"
  "61:S2_byteload_0x1000"
  "64:S2_qwordload_0x1000"
  "5:bg_load_0x1c77"
  "8:bg_load_0x105b"
  "40:bg_load_0x1fe1"
  "44:bg_mul_0x219f"
  "48:bg_load_0x17d8"
)

ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
[ "$ssb" = Vulnerable ] || { echo "ERROR: SSBD not off (spec_store_bypass=$ssb). Won't reproduce."; exit 1; }
if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then ins=( "$F"/min_inputs/input*.bin ); isrc=min_inputs
else ins=( "$F"/input*.bin ); isrc=input; fi
echo "src=minimized.asm  inputs=${#ins[@]} ($isrc)  REPS=$REPS  CLAUSE=[$CLAUSE]  ALIGNED=$ALIGNED"

# sanity: verify each candidate line really is an `and <reg>, 0b...` mask line
echo "--- candidate mask lines ---"
for c in "${CANDS[@]}"; do
  ln="${c%%:*}"; lbl="${c#*:}"
  txt=$(awk -v L="$ln" 'NR==L' "$src")
  printf '  L%-3s %-22s | %s\n' "$ln" "$lbl" "$txt"
  case "$txt" in *"and "*"0b"*) : ;; *) echo "    WARN: L$ln is not an 'and .. 0b..' mask line -> minimized.asm changed? check line numbers"; esac
done
echo

# config: same as the confirmed reproduction (cond-bpas, ct, SSBD off, priming on)
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

# reproduce a given asm REPS times, echo "present/REPS"
run_asm() {
  local asm="$1" tag="$2" real=0 i log viol dur
  for i in $(seq 1 "$REPS"); do
    log="$F/ping_${tag}_$i.log"
    ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
    viol=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
    dur=$(grep -aoE 'Duration: [0-9.]+' "$log" | grep -aoE '[0-9.]+' | tail -1)
    if [ -n "$viol" ] && [ "$viol" -ge 1 ] 2>/dev/null; then real=$((real+1)); fi
    printf '    [%s %s/%s] Violations=%s (%ss)\n' "$tag" "$i" "$REPS" "${viol:-?}" "${dur:-?}"
  done
  echo "$real"
}

echo "### baseline (unmodified minimized.asm) ###"
base=$(run_asm "$src" baseline | tail -1)
echo "  baseline: present $base/$REPS"
if [ "$base" -lt 1 ]; then echo "ERROR: baseline didn't reproduce; abort (flakiness? rerun)."; rm -f "$cfg"; exit 1; fi

declare -A RES
for c in "${CANDS[@]}"; do
  ln="${c%%:*}"; lbl="${c#*:}"
  var="$F/min_align_${lbl}.asm"
  awk -v L="$ln" -v A="$ALIGNED" 'NR==L{ sub(/0b[01]+/, A) } {print}' "$src" > "$var"
  echo; echo "### align only L$ln ($lbl) -> $ALIGNED ###"
  RES["$lbl"]=$(run_asm "$var" "$lbl" | tail -1)
  echo "  $lbl: present ${RES[$lbl]}/$REPS"
done
rm -f "$cfg"

echo; echo "==================== SUMMARY (present / $REPS) ===================="
printf '  %-24s %s   (unmodified)\n' "baseline" "$base"
for c in "${CANDS[@]}"; do
  lbl="${c#*:}"; r="${RES[$lbl]:-?}"
  mark=""; [ "$r" = 0 ] && mark="   <-- alignment KILLS it (critical access)"
  printf '  %-24s %s%s\n' "$lbl" "$r" "$mark"
done
echo "------------------------------------------------------------------"
echo "Reading: a candidate at 0/$REPS means 8B-aligning JUST that access removes the"
echo "leak -> that access's sub-8B offset / partial-overlap is essential. Compare the"
echo "store vs load of a site to tell which side must be unaligned."
echo "variants: $F/min_align_*.asm   logs: $F/ping_*_*.log"
