#!/usr/bin/env bash
# Re-run the minimize pipeline on the comment-stripped gadget (minimized_clean.asm).
# Produces NEW files; leaves minimized.asm and minimized_clean.asm untouched.
#
#   input : violation-*/minimized_clean.asm
#   output: violation-*/minimized2.asm   (+ min_inputs2/)
#
# Fast path (priming OFF) with all passes, cond-bpas, SSBD off, saved inputs.
#
# Env: NI=100  RETRIES=8  CLAUSE='cond, bpas'
# Usage: ./reminimize_clean.sh [violation-folder=violation-260701-234841]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:-violation-260701-234841}"; F="$VD/$name"
NI="${NI:-100}"; RETRIES="${RETRIES:-8}"; CLAUSE="${CLAUSE:-cond, bpas}"
cd "$RV" || exit 1
in="$F/minimized_clean.asm"; [ -f "$in" ] || { echo "ERROR: no minimized_clean.asm in $F (run the strip step first)"; exit 1; }
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
[ "$ssb" = Vulnerable ] || { echo "ERROR: SSBD not off (spec_store_bypass=$ssb)."; exit 1; }
[ -e /sys/rvzr_executor/enable_psfd ] && echo 0 > /sys/rvzr_executor/enable_psfd 2>/dev/null

echo "re-minimizing $in  (NI=$NI RETRIES=$RETRIES CLAUSE=[$CLAUSE], priming OFF)"
mcfg="$(mktemp)"
grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^[[:space:]]*inputs_per_class:|^[[:space:]]*enable_priming:|^[[:space:]]*minimizer_retries:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$mcfg"
grep -E '^(data_generator_seed|inputs_per_class):' "$F/reproduce.yaml" >> "$mcfg"
{ echo "x86_executor_enable_ssbp_patch: false"
  echo "contract_observation_clause: ct"
  echo "contract_execution_clause: [$CLAUSE]"
  echo "minimizer_retries: $RETRIES"
  echo "enable_priming: false"; } >> "$mcfg"

out="$F/minimized2.asm"; mkdir -p "$F/min_inputs2"
export RVZR_MINIMIZE_INPUT_DIR="$F"
./revizor.py minimize -s base.json -c "$mcfg" -t "$in" -o "$out" -i "$NI" \
    --input-outdir "$F/min_inputs2" \
    --enable-instruction-pass true --enable-nop-pass true --enable-constant-pass true \
    --enable-mask-pass true --enable-input-diff-pass true --enable-comment-pass true \
    > "$F/reminimize.log" 2>&1
rc=$?; rm -f "$mcfg"
if [ ! -f "$out" ] || [ "$rc" -ne 0 ]; then
  echo "re-minimize FAILED (rc=$rc). tail:"; grep -avE 'DEBUG: Attempting' "$F/reminimize.log" | tail -12; exit 1
fi

n_in=$(grep -acE 'r14' "$in"); n_out=$(grep -acE 'r14' "$out")
echo "=================================================="
echo "DONE.  mem-ops: minimized_clean=$n_in -> minimized2=$n_out"
echo "  new gadget: $out"
echo "  new inputs: $F/min_inputs2/"
echo "  leaked-byte line(s): $(grep -aE 'Result: Leaked|Addresses:' "$F/reminimize.log" | tail -2 | tr '\n' ' ')"
echo "  log:        $F/reminimize.log"
echo "Verify it survives priming, e.g.:  REPS=5 ./check_ssbd_off.sh $name   # after pointing it at minimized2.asm"
