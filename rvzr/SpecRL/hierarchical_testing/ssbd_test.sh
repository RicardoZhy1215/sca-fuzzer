#!/usr/bin/env bash
# v4 discriminator: reproduce minimized.asm under ct+cond-bpas with SSBD OFF vs ON.
#   SSBD off -> YES (expected)
#   SSBD on  -> NO  => classic Spectre-v4 (SSBD mitigates it)
#            -> YES => survives SSBD => NOT v4  => partial STLF
# Usage: ./ssbd_test.sh <violation-folder>
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:?usage: ./ssbd_test.sh <violation-folder>}"; F="$VD/$name"
cd "$RV" || exit 1
# Use the VERIFIED config: program.asm + original input*.bin (what verify_real
# confirmed YES). minimized.asm can be input-specific (only fires with
# min_inputs) -> unreliable for this toggle.
asm="$F/program.asm"; src="$F/minimize_test.yaml"
[ -f "$src" ] || src="$F/reproduce.yaml"
[ -f "$asm" ] && [ -f "$src" ] || { echo "need program.asm + a config"; exit 1; }
ins=( "$F"/input*.bin )
echo "folder=$name  system spec_store_bypass=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass)"
run() {
  local val="$1" tag="$2"
  local cfg; cfg=$(mktemp)
  grep -v 'x86_executor_enable_ssbp_patch' "$src" > "$cfg"
  echo "x86_executor_enable_ssbp_patch: $val" >> "$cfg"
  local log="$F/ssbd_${tag}.log"
  timeout --signal=INT 300 ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
  rm -f "$cfg"
  local v
  if   grep -aqiE 'no more checkpoints|Traceback|AssertionError' "$log"; then v=ERROR
  elif grep -aqiE 'KeyboardInterrupt' "$log"; then v=TIMEOUT
  elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then v=NO
  elif grep -aqiE 'violation' "$log"; then v=YES
  else v="?"; fi
  printf "  SSBD %-3s (ssbp_patch=%-5s): %s\n" "$tag" "$val" "$v"
}
run false off
run true  on
echo "Interpretation: ON=NO -> Spectre-v4 ; ON=YES -> survives SSBD -> partial STLF (not v4)"
