#!/usr/bin/env bash
# One-shot: minimize a (flaky) violation, then JUDGE whether the minimized
# gadget is REAL (survives full-priming reproduction).
#
# Phase 1 (minimize): drive minimization by the FAST PATH (priming OFF -> ~100%
#   hit, feasible). Uses the fixed cond-bpas contract, SSBD off, and the REAL
#   saved inputs (not seed-regen). -> minimized.asm (+ min_inputs/).
# Phase 2 (verdict): reproduce minimized.asm VERIFY_N times with the FULL
#   pipeline (priming ON). Count how many survive priming (Violations>=1).
#   REAL>=1  -> minimized gadget captures the real signal (confirmed).
#   REAL==0  -> minimize followed fast-path noise (rerun with FAST=0 / more inputs).
#
# Env: NI=100  RETRIES=8  CLAUSE='cond, bpas'  VERIFY_N=8
# Usage: ./minimize_and_verify.sh <violation-folder>
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:?usage: ./minimize_and_verify.sh <folder>}"; F="$VD/$name"
NI="${NI:-100}"; RETRIES="${RETRIES:-8}"; CLAUSE="${CLAUSE:-cond, bpas}"; VERIFY_N="${VERIFY_N:-8}"
cd "$RV" || exit 1
[ -f "$F/program.asm" ] || { echo "no program.asm"; exit 1; }
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
smt=$(cat /sys/devices/system/cpu/smt/active 2>/dev/null)
echo "folder=$name  RETRIES=$RETRIES  CLAUSE=[$CLAUSE]  VERIFY_N=$VERIFY_N  SSBD=$ssb  SMT=$smt"
[ "$ssb" = Vulnerable ] || { echo "ERROR: SSBD not off (Vulnerable); won't reproduce. Set big_fuzz ssbp false."; exit 1; }
[ "$smt" = 1 ] && echo "WARNING: SMT on -> extra P+P noise; consider turning it off."

# build a config: big_fuzz base, strip the keys we set, add ours.  $1 = enable_priming value
mkcfg() {
  local prim="$1"; local out; out="$(mktemp)"
  grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^[[:space:]]*inputs_per_class:|^[[:space:]]*enable_priming:|^[[:space:]]*minimizer_retries:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$out"
  grep -E '^(data_generator_seed|inputs_per_class):' "$F/reproduce.yaml" >> "$out"
  {
    echo "x86_executor_enable_ssbp_patch: false"
    echo "contract_observation_clause: ct"
    echo "contract_execution_clause: [$CLAUSE]"
    echo "minimizer_retries: $RETRIES"
    echo "enable_priming: $prim"
  } >> "$out"
  echo "$out"
}

# ---------------- Phase 1: minimize (fast path) ----------------
echo; echo "################ Phase 1: minimize (priming OFF, fast) ################"
mcfg="$(mkcfg false)"
out="$F/minimized.asm"; mkdir -p "$F/min_inputs"
export RVZR_MINIMIZE_INPUT_DIR="$F"
./revizor.py minimize -s base.json -c "$mcfg" -t "$F/program.asm" -o "$out" -i "$NI" \
    --input-outdir "$F/min_inputs" \
    --enable-instruction-pass true --enable-nop-pass true --enable-constant-pass true \
    --enable-mask-pass true --enable-input-diff-pass true --enable-comment-pass true \
    > "$F/mv_minimize.log" 2>&1
rc=$?; rm -f "$mcfg"
if [ ! -f "$out" ] || [ "$rc" -ne 0 ]; then
  echo "  minimize FAILED (rc=$rc). tail:"; grep -avE 'DEBUG: Attempting' "$F/mv_minimize.log" | tail -8; exit 1
fi
nreal=$(grep -acE 'r14' "$out")
echo "  minimized.asm written ($nreal mem-ops). Leaked-byte line:"
grep -aE 'Result: Leaked|Addresses:' "$F/mv_minimize.log" | tail -2

# ---------------- Phase 2: verdict (full priming) ----------------
echo; echo "################ Phase 2: verdict — does minimized.asm survive priming? ################"
vcfg="$(mkcfg true)"        # priming ON = rigorous
if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then vins=( "$F"/min_inputs/input*.bin ); isrc=min_inputs
else vins=( "$F"/input*.bin ); isrc=input; fi
echo "  verifying minimized.asm with $isrc inputs, $VERIFY_N runs, full pipeline (no timeout)"
real=0; fp=0; oth=0
for i in $(seq 1 "$VERIFY_N"); do
  vlog="$F/mv_verify_$i.log"
  ./revizor.py reproduce -s base.json -c "$vcfg" -t "$out" -i "${vins[@]}" > "$vlog" 2>&1
  viol=$(grep -aoiE '^Violations: [0-9]+' "$vlog" | grep -aoE '[0-9]+' | tail -1)
  prim=$(grep -aoE 'Priming Check: [0-9]+' "$vlog" | grep -aoE '[0-9]+' | tail -1)
  dur=$(grep -aoE 'Duration: [0-9.]+' "$vlog" | grep -aoE '[0-9.]+' | tail -1)
  if   [ -n "$viol" ] && [ "$viol" -ge 1 ] 2>/dev/null; then v=REAL; real=$((real+1))
  elif [ -n "$viol" ] && [ "${prim:-0}" -ge 1 ] 2>/dev/null; then v=FP-priming; fp=$((fp+1))
  else v=OTHER; oth=$((oth+1)); fi
  echo "  [verify $i/$VERIFY_N] -> $v (Violations=${viol:-?}, ${dur:-?}s)"
done
rm -f "$vcfg"
echo "================================================================"
echo "MINIMIZED gadget survives priming: REAL=$real / $VERIFY_N   (FP-priming=$fp, other=$oth)"
if [ "$real" -ge 1 ]; then
  echo "VERDICT: CONFIRMED — minimized.asm reproduces a real (flaky) violation. -> $out"
else
  echo "VERDICT: NOT confirmed — minimized gadget did not survive priming in $VERIFY_N runs."
  echo "         Likely minimized against fast-path noise. Retry: FAST=0 (rigorous) or larger VERIFY_N/NI."
fi
