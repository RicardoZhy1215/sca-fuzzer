#!/usr/bin/env bash
# run_inject2_clean_ab.sh — with the generic-transient preamble removed, is there REAL injection?
# Both random preambles gone (fixed aligned jmps), disclosure still random, P+P, no shared memory.
# Does NOT touch the module (load the clean 1CD139EF yourself).
#   with-attacker > 0, no-attacker == 0 -> GENUINE cross-guest BTB injection.
#   both == 0                            -> clean, but injection is frozen (fixed alignment doesn't collide).
# NOTE: with the preamble gone, most TCs reach full analysis => SLOW (~2-3s/TC). Keep N modest.
# Usage:  N=100 bash run_inject2_clean_ab.sh    (default 100)
set -uo pipefail
REPO=/home/mluo/sca-fuzzer; CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}; BASE=$REPO/base.json; CONFIG=$CAMP/config-v2v-pp.yaml
N=${N:-100}; I=${I:-50}
echo "loaded executor: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
[ -e /sys/rvzr_executor ] || { echo "no /sys/rvzr_executor — load the module first"; exit 2; }
run_one(){ echo "=== [$1] $(date +%H:%M:%S)  (N=$N x I=$I) ==="; mkdir -p "$3"; cd "$REPO"
  "$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$2" -n "$N" -i "$I" -w "$3"
  shopt -s nullglob; local V=("$3"/violation-*); echo " [$1] violations: ${#V[@]}"; eval "$4=${#V[@]}"; }
run_one "WITH-attacker(clean)" "$CAMP/template-v2v-inject2-clean.asm"          "$CAMP/out-inject2-clean"        AV
run_one "NO-attacker(clean)"   "$CAMP/template-v2v-inject2-clean-noattack.asm" "$CAMP/out-inject2-clean-noatk" BV
echo "=================================================================="
echo " RESULT (N=$N):  WITH-attacker=$AV   NO-attacker=$BV"
if   [ "$AV" -gt 0 ] && [ "$BV" -eq 0 ]; then echo " => attacker NECESSARY ==> GENUINE cross-guest BTB INJECTION ✅"
elif [ "$AV" -eq 0 ] && [ "$BV" -eq 0 ]; then echo " => clean, no generic transient — but injection is FROZEN (no BTB collision at fixed alignment)"
else echo " => still leaks without attacker — some generic transient remains (not just the preamble)"; fi
echo "=================================================================="
