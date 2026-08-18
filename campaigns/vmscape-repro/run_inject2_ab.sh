#!/usr/bin/env bash
# run_inject2_ab.sh — is V2V injection v2 (disclosure at BTB-mispredict target) REAL injection?
# A/B: with-attacker vs no-attacker, P+P channel, no shared memory. Does NOT touch the module —
# load the build you want first (clean 1CD139EF).
#   with-attacker leaks, no-attacker == 0 -> genuine cross-guest BTB injection.
#   both leak                             -> still a generic transient (attacker irrelevant).
# Usage:  bash run_inject2_ab.sh     |     N=1000 bash run_inject2_ab.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer; CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}; BASE=$REPO/base.json; CONFIG=$CAMP/config-v2v-pp.yaml
N=${N:-500}; I=${I:-50}
echo "loaded executor: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
[ -e /sys/rvzr_executor ] || { echo "no /sys/rvzr_executor — load the module first"; exit 2; }
run_one(){ echo "=== [$1] $2 (N=$N x I=$I) $(date +%H:%M:%S) ==="; mkdir -p "$3"; cd "$REPO"
  "$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$2" -n "$N" -i "$I" -w "$3"
  shopt -s nullglob; local V=("$3"/violation-*); echo " [$1] violations: ${#V[@]}"; eval "$4=${#V[@]}"; }
run_one "WITH-attacker" "$CAMP/template-v2v-inject2.asm"          "$CAMP/out-inject2"          AV
run_one "NO-attacker"   "$CAMP/template-v2v-inject2-noattack.asm" "$CAMP/out-inject2-noattack" BV
echo "=================================================================="
echo " RESULT (N=$N):  WITH-attacker=$AV   NO-attacker=$BV"
if [ "$AV" -gt 0 ] && [ "$BV" -eq 0 ]; then echo " => attacker NECESSARY ==> GENUINE cross-guest BTB INJECTION"
elif [ "$AV" -gt 0 ] && [ "$BV" -gt 0 ]; then echo " => attacker irrelevant ==> still a generic transient, NOT injection"
else echo " => WITH-attacker didn't leak at N=$N — bump N."; fi
echo "=================================================================="
