#!/usr/bin/env bash
# Minimize a FLAKY violation. Full-priming reproduce-checks are too slow+flaky
# (~40min, 15% hit) to drive minimization, so by default we minimize against the
# FAST PATH (priming OFF -> ~100% hit, minutes/check). The fast-path "violation"
# includes priming-noise, so ALWAYS verify the minimized gadget afterwards with:
#     ./repeat_reproduce.sh <folder>       (on the minimized program)
#
# Env: FAST=1 (default; priming off, fast) | FAST=0 (full pipeline, rigorous but
#      likely infeasible for a flaky case).  RETRIES=<n> minimizer_retries (default 8).
#      CLAUSE='cond, bpas' (default).
# Usage: ./minimize_flaky.sh <violation-folder> [num_inputs]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:?usage: ./minimize_flaky.sh <folder> [num_inputs]}"; F="$VD/$name"; NI="${2:-100}"
FAST="${FAST:-1}"; RETRIES="${RETRIES:-8}"; CLAUSE="${CLAUSE:-cond, bpas}"
cd "$RV" || exit 1
[ -f "$F/program.asm" ] || { echo "no program.asm"; exit 1; }
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
smt=$(cat /sys/devices/system/cpu/smt/active 2>/dev/null)
echo "folder=$name  FAST=$FAST  RETRIES=$RETRIES  CLAUSE=[$CLAUSE]  SSBD=$ssb  SMT=$smt"
[ "$ssb" = Vulnerable ] || echo "WARNING: SSBD not off -> won't reproduce"
[ "$smt" = 1 ] && echo "WARNING: SMT on -> extra noise; consider turning it off"

cfg="$F/minimize_flaky.yaml"
grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^[[:space:]]*inputs_per_class:|^[[:space:]]*enable_priming:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$cfg"
grep -E '^(data_generator_seed|inputs_per_class):' "$F/reproduce.yaml" >> "$cfg"
{
  echo "x86_executor_enable_ssbp_patch: false"
  echo "contract_observation_clause: ct"
  echo "contract_execution_clause: [$CLAUSE]"
  echo "minimizer_retries: $RETRIES"
  [ "$FAST" = 1 ] && echo "enable_priming: false"   # fast-path-driven checks
} >> "$cfg"
echo "--- effective config ---"; grep -E 'seed|inputs_per_class|contract_|ssbp|minimizer_retries|enable_priming|cond|bpas' "$cfg"

out="$F/minimized.asm"; mkdir -p "$F/min_inputs"
export RVZR_MINIMIZE_INPUT_DIR="$F"   # load the real violating inputs (not seed-regen)
echo "=== running minimize (this can take a while) ==="
./revizor.py minimize -s base.json -c "$cfg" -t "$F/program.asm" -o "$out" -i "$NI" \
    --input-outdir "$F/min_inputs" \
    --enable-instruction-pass true --enable-nop-pass true --enable-constant-pass true \
    --enable-mask-pass true --enable-input-diff-pass true --enable-comment-pass true
rc=$?
echo "=== exit=$rc ; minimized gadget -> $out ==="
[ -f "$out" ] && grep -vE '^\s*$|^\.section|^\.intel|^\.function|^\.test_case_exit' "$out"
echo
echo ">>> NEXT: verify the minimized gadget actually survives priming (it was minimized"
echo ">>>       against the fast path, which includes noise):"
echo ">>>   cp $out $F/program_minimized.asm   # or point repeat_reproduce at $out"
echo ">>>   then run repeat_reproduce on the minimized program and check REAL>0"
