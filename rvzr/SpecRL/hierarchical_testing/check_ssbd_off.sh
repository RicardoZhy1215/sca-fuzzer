#!/usr/bin/env bash
# Confirm whether minimized.asm still triggers a violation under SMT OFF + SSBD OFF.
#
# SMT off  = removes SMT/Prime+Probe noise (cleanest measurement).
# SSBD off = the vulnerable condition (store-bypass window open).
# Expectation: the violation REPRODUCES (Violations >= 1).
#
# Env: REPS=5  CLAUSE='cond, bpas'
# Usage: ./check_ssbd_off.sh [violation-folder=violation-260701-234841]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:-violation-260701-234841}"; F="$VD/$name"
REPS="${REPS:-5}"; CLAUSE="${CLAUSE:-cond, bpas}"
cd "$RV" || exit 1
asm="$F/minimized.asm"; [ -f "$asm" ] || { echo "ERROR: no minimized.asm in $F"; exit 1; }

# --- require SMT off ---
smt=$(cat /sys/devices/system/cpu/smt/active 2>/dev/null)
echo "SMT active = ${smt:-unknown}   (want 0 = off)"
if [ "$smt" = 1 ]; then
  echo "ERROR: SMT is ON. Turn it off first, then rerun:"
  echo "    echo off | sudo tee /sys/devices/system/cpu/smt/control"
  exit 1
fi

# --- inputs (prefer minimized inputs) ---
if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then ins=( "$F"/min_inputs/input*.bin ); isrc=min_inputs
else ins=( "$F"/input*.bin ); isrc=input; fi

# --- force the PSFD knob off (clean baseline), if the rebuilt executor exposes it ---
[ -e /sys/rvzr_executor/enable_psfd ] && echo 0 > /sys/rvzr_executor/enable_psfd 2>/dev/null

sys_ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
echo "system spec_store_bypass = ${sys_ssb:-?}   (executor SSBD forced OFF per-run via the yaml below)"
echo "asm=$(basename "$asm")  inputs=${#ins[@]} ($isrc)  REPS=$REPS  CLAUSE=[$CLAUSE]  ==> SSBD = OFF"

# --- config: SSBD OFF (vulnerable condition) ---
cfg="$(mktemp)"
grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^[[:space:]]*inputs_per_class:|^[[:space:]]*enable_priming:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$cfg"
grep -E '^(data_generator_seed|inputs_per_class):' "$F/reproduce.yaml" >> "$cfg"
{
  echo "x86_executor_enable_ssbp_patch: false"    # <-- SSBD OFF (vulnerable)
  echo "contract_observation_clause: ct"
  echo "contract_execution_clause: [$CLAUSE]"
  echo "enable_priming: true"
} >> "$cfg"

real=0; err=0
for i in $(seq 1 "$REPS"); do
  log="$F/ssbdoff_$i.log"
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
echo "SMT off + SSBD OFF:  violation present in $real / $REPS runs   (errors: $err)"
if [ "$real" -eq "$REPS" ] && [ "$err" -eq 0 ]; then
  echo "VERDICT: reproduces every run ($real/$REPS) — robust under SMT off + SSBD off."
elif [ "$real" -ge 1 ]; then
  echo "VERDICT: reproduces $real/$REPS (some misses = residual flakiness)."
else
  echo "VERDICT: did NOT reproduce (0/$REPS) — unexpected under SSBD off; check logs / flakiness."
fi
echo "logs: $F/ssbdoff_*.log"
