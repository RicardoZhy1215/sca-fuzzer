#!/usr/bin/env bash
# Is the `mul` causal, or just address plumbing? Reproduce program.asm and two
# swapped variants (ct+cond-bpas, SSBD off, original inputs).
#   imul rax,[mem] : keep multiply, drop the rdx (high-product) write
#   mov  rax,[mem] : drop multiply (rdx left unchanged)
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:?usage: ./swap_mul.sh <folder>}"; F="$VD/$name"
mulline="${2:-mul qword ptr \[r14 + rcx\]}"
cd "$RV" || exit 1
src="$F/minimize_test.yaml"; [ -f "$src" ] || src="$F/reproduce.yaml"
cfg=$(mktemp)
grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^  - (seq|bpas|cond|cond-bpas)$' "$src" > "$cfg"
grep -qE 'contract_observation_clause' "$cfg" || echo 'contract_observation_clause: ct' >> "$cfg"
printf 'x86_executor_enable_ssbp_patch: false\ncontract_execution_clause: [cond, bpas]\n' >> "$cfg"
ins=( "$F"/input*.bin )
echo "folder=$name  SSBD(system)=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass)"
repro() {
  local asm="$1"; local tag="$2"; local log="$F/swap_${tag}.log"
  timeout --signal=INT 300 ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
  local rc=$?
  if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then echo TIMEOUT
  elif grep -aqiE 'no more checkpoints|Traceback|AssertionError' "$log"; then echo ERROR
  elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then echo NO
  elif grep -aqiE 'violation' "$log"; then echo YES; else echo "?"; fi
}
sed "s/${mulline}/imul rax, qword ptr [r14 + rcx]/" "$F/program.asm" > "$F/swap_imul.asm"
sed "s/${mulline}/mov rax, qword ptr [r14 + rcx]/"  "$F/program.asm" > "$F/swap_mov.asm"
echo "  swap_imul changed lines: $(diff "$F/program.asm" "$F/swap_imul.asm" | grep -c '^>')"
echo "  swap_mov  changed lines: $(diff "$F/program.asm" "$F/swap_mov.asm"  | grep -c '^>')"
b=$(repro "$F/program.asm" baseline)
im=$(repro "$F/swap_imul.asm" imul)
mv=$(repro "$F/swap_mov.asm" mov)
rm -f "$cfg"
echo "  baseline (mul [mem])        : $b"
echo "  imul rax,[mem] (no rdx wr)  : $im"
echo "  mov  rax,[mem] (no multiply): $mv"
echo "Interpret: im=NO -> rdx high-product needed | im=YES&mv=NO -> multiply(product) causal | mv=YES -> mul was just plumbing(load suffices)"
