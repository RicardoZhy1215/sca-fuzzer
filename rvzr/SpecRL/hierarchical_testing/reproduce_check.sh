#!/usr/bin/env bash
# Reproduce-check for a single violation folder using its own reproduce.yaml.
#
# The per-folder reproduce.yaml is the exact config captured at detection time
# (here: ct + delayed-exception-handling, SSBD ON, faulty-page actor, filters),
# so we replay with THAT plus the saved input_*.bin. No sudo needed: the
# rvzr_executor sysfs is world-writable and the KM is already loaded.
#
# Usage:
#   bash reproduce_check.sh                       # defaults to violation-260721-212748
#   bash reproduce_check.sh <violation-folder>    # any other folder
#   TIMEOUT=900 bash reproduce_check.sh <folder>  # override wall-clock limit (s)
set -u

REPO="/home/hz25d/beyond_v1_v4/sca-fuzzer"
PY="/home/hz25d/miniconda3/envs/specrl/bin/python"
HT="rvzr/SpecRL/hierarchical_testing"                 # relative to REPO
NAME="${1:-violation-260721-212748}"
TIMEOUT="${TIMEOUT:-600}"

V="$HT/$NAME"
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }
for f in "$V/reproduce.yaml" "$V/program.asm" base.json; do
    [ -f "$f" ] || { echo "missing: $REPO/$f"; exit 1; }
done

shopt -s nullglob
inputs=( "$V"/input_*.bin )
shopt -u nullglob
[ ${#inputs[@]} -gt 0 ] || { echo "no input_*.bin in $V"; exit 1; }

log="$V/reproduce_check.log"
echo "folder   : $NAME"
echo "contract : from $V/reproduce.yaml (ct + delayed-exception-handling, SSBD ON)"
echo "inputs   : ${#inputs[@]} saved input_*.bin"
echo "log      : $REPO/$log   (timeout ${TIMEOUT}s)"
echo "SSBD(sys): $(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)"
echo "------------------------------------------------------------"

timeout --signal=INT "$TIMEOUT" \
    "$PY" ./revizor.py reproduce \
        -s base.json \
        -c "$V/reproduce.yaml" \
        -t "$V/program.asm" \
        -i "${inputs[@]}" \
    2>&1 | tee "$log"
rc=${PIPESTATUS[0]}

echo "------------------------------------------------------------"
nviol="$(grep -aoiE 'violations?: *[0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)"
if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then res="TIMEOUT (inconclusive)"
elif grep -aqiE 'Traceback|AssertionError|no more checkpoints' "$log"; then res="ERROR (see log tail)"
elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then res="NO  (does NOT reproduce)"
elif grep -aqiE 'violation' "$log"; then res="YES (REPRODUCED)"
else res="? (see log)"; fi
echo "RESULT: $res   [n_violations=${nviol:-?}]  rc=$rc"
[ "${res:0:3}" = "ERR" ] && { echo '--- last 20 log lines ---'; tail -20 "$log"; }
