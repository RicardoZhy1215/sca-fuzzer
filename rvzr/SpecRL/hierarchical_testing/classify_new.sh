#!/usr/bin/env bash
# Reproduce + classify ONE (post-fix) violation. Explicit contracts, SSBD off,
# program.asm + saved input*.bin, full pipeline.
#   cond-bpas (FIXED) : YES = real violation under the contract that found it
#   bpas              : NO = plain v4 (store-bypass explains it) ;
#                       YES = beyond store-bypass (genuinely interesting)
#   seq               : baseline (any speculative leak -> YES)
# Usage: ./classify_new.sh <violation-folder>
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:?usage: ./classify_new.sh <violation-folder>}"; F="$VD/$name"
cd "$RV" || exit 1
[ -f "$F/program.asm" ] || { echo "no program.asm in $F"; exit 1; }
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
echo "folder=$name  SSBD(system)=$ssb"
[ "$ssb" = Vulnerable ] || echo "WARNING: SSBD not off -> store-bypass leaks won't appear (results trivially NO)"
pgrep -af "train.py" | grep -qv pgrep && echo "WARNING: train.py is RUNNING -> executor contention! stop training first."
ins=( "$F"/input*.bin )
run() {
  local clause="$1"; local tag="$2"; local cfg; cfg=$(mktemp)
  grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$cfg"
  grep -E '^(data_generator_seed|inputs_per_class):' "$F/reproduce.yaml" >> "$cfg"
  printf 'x86_executor_enable_ssbp_patch: false\ncontract_observation_clause: ct\ncontract_execution_clause: [%s]\n' "$clause" >> "$cfg"
  local log="$F/classify_${tag}.log"
  timeout --signal=INT 300 ./revizor.py reproduce -s base.json -c "$cfg" -t "$F/program.asm" -i "${ins[@]}" > "$log" 2>&1
  local rc=$?; rm -f "$cfg"
  local v
  if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then v=TIMEOUT
  elif grep -aqiE 'no more checkpoints|Traceback|AssertionError' "$log"; then v=ERROR
  elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then v=NO
  elif grep -aqiE 'violation' "$log"; then v=YES; else v="?"; fi
  printf "  %-10s : %s\n" "$tag" "$v"
}
run "cond, bpas" cond-bpas
run "bpas"       bpas
run "seq"        seq
echo "Read: cond-bpas=YES -> real | bpas=YES -> beyond store-bypass(interesting) | bpas=NO -> plain v4"
