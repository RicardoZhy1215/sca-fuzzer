#!/usr/bin/env bash
# run_inject_clean.sh — the CLEAN V2V injection (no random_instructions, WITH trainer).
# vm trains BTB (jmp rax) -> vmvictim should mispredict (jmp rbx) into the disclosure.
# P+P channel, no shared memory. Does NOT touch the module (load clean 1CD139EF yourself).
#   0 violations => FROZEN: fixed alignment doesn't collide in the BTB, no injection (expected).
#   >0           => the clean injection actually fires (would be a real surprise).
# Usage:  bash run_inject_clean.sh   |   N=1000 bash run_inject_clean.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer; CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}; BASE=$REPO/base.json
CONFIG=$CAMP/config-v2v-pp.yaml
TEMPLATE=$CAMP/template-v2v-inject.asm
OUT=$CAMP/out-v2v-inject-clean
N=${N:-500}; I=${I:-50}
echo "=================================================================="
echo " CLEAN V2V injection (no random_instructions, WITH trainer)  N=$N x I=$I"
echo " template: $TEMPLATE   |   config: $CONFIG (P+P, no shared mem)"
echo "=================================================================="
echo "loaded executor: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
[ -e /sys/rvzr_executor ] || { echo "no /sys/rvzr_executor — load the module first"; exit 2; }
mkdir -p "$OUT"; cd "$REPO"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT"
shopt -s nullglob; V=("$OUT"/violation-*)
echo "=================================================================="
echo " CLEAN injection (with trainer) => violations: ${#V[@]}"
[ "${#V[@]}" -eq 0 ] && echo "  => FROZEN as expected (no BTB collision at fixed alignment; clean templates never leaked)"
[ "${#V[@]}" -gt 0 ] && echo "  => it FIRED — unexpected; worth digging into"
echo "=================================================================="
