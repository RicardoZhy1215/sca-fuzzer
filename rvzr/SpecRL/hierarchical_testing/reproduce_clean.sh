#!/usr/bin/env bash
# Reproduce the comment-stripped gadget (minimized_clean.asm) to confirm removing
# the "# mem access" comments did not change behavior (it shouldn't — the asm
# parser ignores comments). Vulnerable condition: SSBD off, cond-bpas, ct, priming on.
#
# Env: REPS=5  CLAUSE='cond, bpas'  ASM=minimized_clean.asm
# Usage: ./reproduce_clean.sh [violation-folder=violation-260701-234841]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:-violation-260701-234841}"; F="$VD/$name"
REPS="${REPS:-5}"; CLAUSE="${CLAUSE:-cond, bpas}"; ASM="${ASM:-minimized_clean.asm}"
cd "$RV" || exit 1
asm="$F/$ASM"; [ -f "$asm" ] || { echo "ERROR: no $ASM in $F"; exit 1; }
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
[ "$ssb" = Vulnerable ] || { echo "ERROR: SSBD not off (spec_store_bypass=$ssb)."; exit 1; }
smt=$(cat /sys/devices/system/cpu/smt/active 2>/dev/null); [ "$smt" = 1 ] && echo "WARNING: SMT on -> extra noise."
[ -e /sys/rvzr_executor/enable_psfd ] && echo 0 > /sys/rvzr_executor/enable_psfd 2>/dev/null
if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then ins=( "$F"/min_inputs/input*.bin ); isrc=min_inputs
else ins=( "$F"/input*.bin ); isrc=input; fi
echo "asm=$ASM  inputs=${#ins[@]} ($isrc)  REPS=$REPS  CLAUSE=[$CLAUSE]  SSBD=off"

cfg="$(mktemp)"
grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^[[:space:]]*inputs_per_class:|^[[:space:]]*enable_priming:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$cfg"
grep -E '^(data_generator_seed|inputs_per_class):' "$F/reproduce.yaml" >> "$cfg"
{ echo "x86_executor_enable_ssbp_patch: false"
  echo "contract_observation_clause: ct"
  echo "contract_execution_clause: [$CLAUSE]"
  echo "enable_priming: true"; } >> "$cfg"

real=0; err=0
for i in $(seq 1 "$REPS"); do
  log="$F/reproclean_$i.log"
  ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
  if grep -aqiE 'operand type mismatch|Error appeared while assembling|Traceback|Segmentation' "$log"; then
    err=$((err+1)); echo "  [run $i/$REPS] ERROR (see $log)"; continue
  fi
  viol=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
  dur=$(grep -aoE 'Duration: [0-9.]+' "$log" | grep -aoE '[0-9.]+' | tail -1)
  [ -n "$viol" ] && [ "$viol" -ge 1 ] 2>/dev/null && real=$((real+1))
  echo "  [run $i/$REPS] Violations=${viol:-?} (${dur:-?}s)"
done
rm -f "$cfg"

echo "=================================================="
echo "$ASM :  violation present in $real / $REPS runs   (errors: $err)"
if [ "$real" -ge 1 ] && [ "$err" -eq 0 ]; then
  echo "VERDICT: reproduces -> stripping the # mem access comments did NOT change behavior. OK to re-minimize."
elif [ "$err" -gt 0 ]; then
  echo "VERDICT: build/run error -> the clean file is malformed; inspect $F/reproclean_*.log"
else
  echo "VERDICT: did NOT reproduce (0/$REPS) -> unexpected; compare against minimized.asm."
fi
echo "logs: $F/reproclean_*.log"
