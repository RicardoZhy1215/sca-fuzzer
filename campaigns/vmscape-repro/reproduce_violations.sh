#!/usr/bin/env bash
#
# reproduce_violations.sh — replay saved VMSCAPE violations on hardware and check
# they still reproduce.  Uses Revizor's `reproduce` mode:
#
#   rvzr reproduce -s base.json -c <dir>/reproduce.yaml -t <dir>/program.asm -i <dir>/input_*.bin
#
# (see docs/howto/root-cause-a-violation.md — a reproduction prints
#  "================ Violations detected ================").
#
# By default it replays the two TC-585 violations: guest-kernel and guest-user,
# which are the exact same program under the two privilege configs.
#
# Usage:
#   bash reproduce_violations.sh                      # both default violations
#   bash reproduce_violations.sh <violation-dir> ...  # specific dirs
#   NIN=20 bash reproduce_violations.sh               # only first NIN inputs (faster)
#
set -uo pipefail

REPO=/home/mluo/sca-fuzzer
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
OUT=$REPO/campaigns/vmscape-repro/out
KO=$REPO/rvzr/executor_km/rvzr_executor.ko
WANT_SRCVER=$(modinfo -F srcversion "$KO" 2>/dev/null || echo unknown)
LOGDIR=${LOGDIR:-$OUT/reproduce-logs}
NIN=${NIN:-0}                                # 0 = all saved inputs
mkdir -p "$LOGDIR"

# default targets: the two same-program TC-585 violations
VIOLATIONS=(
  "guest-kernel:$OUT/violation-260806-230651"
  "guest-user:$OUT/violation-260807-055254"
)
if [ "$#" -gt 0 ]; then
    VIOLATIONS=(); for d in "$@"; do VIOLATIONS+=("$(basename "$d"):${d%/}"); done
fi

# --- environment: SMT off + correct module -----------------------------------
echo "=== environment ==="
SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo n/a)
echo "SMT = $SMT (want off)"
[ "$SMT" = on ] && echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null 2>&1 && echo "  -> disabled"
LOADED=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null || echo none)
echo "module srcversion = $LOADED (want $WANT_SRCVER)"
if [ "$LOADED" != "$WANT_SRCVER" ]; then
    echo "  -> loading guest-user build (sudo)"
    lsmod | grep -qw rvzr_executor && sudo rmmod rvzr_executor
    sudo insmod "$KO" || { echo "insmod FAILED"; exit 2; }
fi
[ -e /sys/rvzr_executor ] || { echo "/sys/rvzr_executor missing — module not initialized"; exit 2; }
echo

# --- reproduce each ----------------------------------------------------------
PASS=0; FAIL=0
for entry in "${VIOLATIONS[@]}"; do
    label=${entry%%:*}; V=${entry#*:}
    [ -f "$V/program.asm" ] && [ -f "$V/reproduce.yaml" ] || { echo "[$label] missing program/reproduce.yaml — skip"; continue; }
    mapfile -t allin < <(ls "$V"/input_*.bin 2>/dev/null | sort)
    [ "${#allin[@]}" -eq 0 ] && { echo "[$label] no inputs — skip"; continue; }
    if [ "$NIN" -gt 0 ] && [ "$NIN" -lt "${#allin[@]}" ]; then
        ins=("${allin[@]:0:$NIN}")
    else
        ins=("${allin[@]}")
    fi
    LOG="$LOGDIR/$label.log"

    echo "=================================================================="
    echo "[$label]  $V   (${#ins[@]}/${#allin[@]} inputs)"
    cd "$REPO"
    "$PY" revizor.py reproduce -s "$BASE" -c "$V/reproduce.yaml" \
        -t "$V/program.asm" -i "${ins[@]}" > "$LOG" 2>&1
    rc=$?
    if grep -qiE "Violations detected|Violations: [1-9]" "$LOG"; then
        echo "  => REPRODUCED  ✅   (violation still present)"
        PASS=$((PASS+1))
    else
        echo "  => clean (no violation)   rc=$rc"
        FAIL=$((FAIL+1))
    fi
    grep -iE "Violations:|Duration|Increasing sample|Test Cases:" "$LOG" | tail -4 | sed 's/^/       /'
done

echo "=================================================================="
echo " reproduced: $PASS    clean: $FAIL    (full logs in $LOGDIR)"
echo "=================================================================="
[ "$PASS" -gt 0 ] && exit 0 || exit 1
