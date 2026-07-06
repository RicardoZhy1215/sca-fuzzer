#!/usr/bin/env bash
# Minimize ONE confirmed-real violation to its minimal gadget (reveals partial
# STLF: STORE [A] then partially-overlapping LOAD).  Modeled on the reference
# invocation (per-folder minimize_test.yaml + input-diff pass + input-outdir),
# but loads the EXACT saved inputs (seed-regen does NOT reproduce RL-generated
# violations here, since the RL input RNG advances across episodes).
#
# Usage:  ./minimize_one.sh <violation-folder> [num_inputs]
#   CLAUSE='cond, bpas' (default) | 'bpas' | 'seq'
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:?usage: ./minimize_one.sh <violation-folder> [num_inputs]}"
F="$VD/$name"; NI="${2:-100}"; CLAUSE="${CLAUSE:-cond, bpas}"
cd "$RV" || exit 1
[ -f "$F/program.asm" ] || { echo "no program.asm in $F"; exit 1; }
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
echo "SSBD(system)=$ssb  CLAUSE=[$CLAUSE]  num_inputs=$NI  folder=$name"
[ "$ssb" = "Vulnerable" ] || echo "WARNING: SSBD not off -> violation may not reproduce."

# --- minimize_test.yaml: respect an existing one; else generate it ---
# SSBD: default false (off) so store-bypass leaks reproduce in standalone
# minimize. Set SSBP_ON=1 to keep SSBD ON -> discriminating test: if the
# violation still reproduces with SSBD on, it survives SSBD => NOT classic v4
# (store-bypass) => evidence for partial STLF.
cfg="$F/minimize_test.yaml"
if [ -f "$cfg" ]; then
  echo "using EXISTING $cfg (not overwriting)"
else
  ssbp_val=$([ "${SSBP_ON:-0}" = 1 ] && echo true || echo false)
  grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^[[:space:]]*inputs_per_class:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$cfg"
  grep -E '^(data_generator_seed|inputs_per_class):' "$F/reproduce.yaml" >> "$cfg"
  {
    echo "x86_executor_enable_ssbp_patch: $ssbp_val"
    echo "contract_observation_clause: ct"
    echo "contract_execution_clause: [$CLAUSE]"
    echo "minimizer_retries: 5"
  } >> "$cfg"
fi
echo "--- minimize_test.yaml (contract/seed/ipc/ssbp) ---"
grep -E 'data_generator_seed|inputs_per_class|contract_|cond|bpas|seq|ssbp' "$cfg"

out="$F/minimized.asm"
mkdir -p "$F/min_inputs"
# load the EXACT violating inputs (see header)
export RVZR_MINIMIZE_INPUT_DIR="$F"

./revizor.py minimize -s base.json -c "$cfg" -t "$F/program.asm" -o "$out" -i "$NI" \
    --input-outdir "$F/min_inputs" \
    --enable-instruction-pass true \
    --enable-nop-pass         true \
    --enable-constant-pass    true \
    --enable-mask-pass        true \
    --enable-input-diff-pass  true \
    --enable-comment-pass     true
rc=$?
echo "=== exit=$rc ; minimized gadget -> $out ==="
[ -f "$out" ] && grep -vE '^\s*$|^\.section|^\.intel|^\.function|^\.test_case_exit' "$out"
