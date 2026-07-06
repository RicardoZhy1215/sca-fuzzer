#!/usr/bin/env bash
# Verify the MINIMIZED test case still triggers the violation.
# Replays minimized.asm with the minimized inputs (min_inputs/, from the
# input-diff pass) under the SAME config minimize used (minimize_test.yaml:
# ct + cond-bpas + the ssbp setting you minimized with).
#
# Usage:  ./verify_minimized.sh <violation-folder>
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:?usage: ./verify_minimized.sh <violation-folder>}"
F="$VD/$name"
cd "$RV" || exit 1

asm="$F/minimized.asm"
cfg="$F/minimize_test.yaml"
[ -f "$asm" ] || { echo "missing $asm (run minimize first)"; exit 1; }
[ -f "$cfg" ] || { echo "missing $cfg (run minimize first)"; exit 1; }

if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then
  inputs=( "$F"/min_inputs/input*.bin ); src="min_inputs"
else
  inputs=( "$F"/input*.bin ); src="original input*.bin"
fi

ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
echo "folder=$name | inputs=${#inputs[@]} ($src) | SSBD(system)=$ssb"
echo "--- minimize_test.yaml (contract/ssbp) ---"
grep -E 'contract_|cond|bpas|seq|ssbp|inputs_per_class|seed' "$cfg"
echo "------------------------------------------------------------"

log="$F/verify_minimized.log"
./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${inputs[@]}" > "$log" 2>&1
rc=$?
nv=$(grep -aoiE 'violations?: *([0-9]+)' "$log" | grep -aoE '[0-9]+' | tail -1)
if   grep -aqiE 'no more checkpoints|Traceback|AssertionError' "$log"; then v="ERROR (crash)"
elif grep -aqiE 'KeyboardInterrupt' "$log" || [ "$rc" -eq 124 ];        then v="TIMEOUT"
elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then v="NO  (minimized asm does NOT trigger)"
elif grep -aqiE 'violation' "$log";                                      then v="YES (minimized asm STILL triggers)"
else v="? (see log)"; fi
echo "RESULT: $v   [n_violations=${nv:-?}]   log: $log"
[ "${v:0:3}" = "ERR" ] && tail -15 "$log"
