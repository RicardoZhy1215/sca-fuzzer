#!/usr/bin/env bash
#
# verify_125257.sh — reproduce the minimized violation-260817-125257 program 10 times, report the rate.
#
# Runs are foreground and sequential (one reproduce fully exits before the next starts).
# exit 1 = violation reproduced, 0 = clean.
#
# Usage:
#   bash verify_125257.sh              # N=10 on program.min.asm
#   N=20 bash verify_125257.sh
#   PROG=$V/program.asm bash verify_125257.sh    # the original (un-minimized) program instead
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-125257}
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/reproduce.yaml
PROG=${PROG:-$V/program.min.asm}
N=${N:-10}

count_instr () { grep -vE '^\s*(#|\.|$)' "$1" 2>/dev/null | grep -cvE '^\s*$'; }
[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

echo "=================================================================="
echo " VERIFY violation-260817-125257  — reproduce x$N"
echo "   program : $PROG  ($(count_instr "$PROG") instrs)"
echo "   config  : $CONFIG   inputs: $(ls "$V"/input_*.bin|wc -l)   module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "=================================================================="
cd "$REPO"

hits=0
for r in $(seq 1 "$N"); do
    "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$PROG" -i "$V"/input_*.bin >/dev/null 2>&1
    rc=$?
    [ "$rc" -eq 1 ] && hits=$((hits+1))
    printf "  run %2d: exit=%d  %s\n" "$r" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
done

echo "=================================================================="
echo " RESULT: reproduced $hits/$N"
[ "$hits" -gt 0 ] && echo " => the minimized program still triggers the violation." \
                  || echo " => did not reproduce this round (raise N, confirm perf_env.sh / clean module)."
echo "=================================================================="