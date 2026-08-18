#!/usr/bin/env bash
#
# run_inject_train.sh — CLEAN V2V injection with STRONG (looped) BTB training.
#
# Tests the hypothesis: the clean single-jmp injection (template-v2v-inject.asm / inject2) stays
# frozen because ONE training jmp can't overpower the victim's own accumulated "safe" BTB history.
# Here the attacker trains the SAME jmp PC 64x in a tight loop (jmp -> vm_disc -> dec r8; jnz back)
# before the victim runs. No random_instructions — the only moving part vs the frozen clean template
# is training strength. Alignment is verified byte-exact (attacker jmp==victim jmp @0x46; trained
# target vm_disc==victim disclosure @0x56).
#
#   >0 violations => single-jmp training WAS the blocker => genuine cross-domain injection reproduces.
#   0  violations => training strength is NOT the blocker => injection still doesn't land clean.
#
# Channel: P+P (config-v2v-pp.yaml), NO shared memory — the realistic vmscape channel.
# This script does NOT touch the kernel module; load the clean alias-free build (1CD139EF) first:
#   sudo rmmod rvzr_executor; sudo insmod rvzr/executor_km/rvzr_executor.ko
#
# Usage:
#   bash run_inject_train.sh          # N=2000 x I=50
#   N=500 bash run_inject_train.sh    # quicker
#   SMOKE=1 bash run_inject_train.sh  # 3-TC pipeline check
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
CONFIG=$CAMP/config-v2v-pp.yaml
TEMPLATE=$CAMP/template-v2v-inject-train.asm
OUT=$CAMP/out-inject-train
N=${N:-2000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=3; I=10; }

echo "=================================================================="
echo " V2V INJECTION + STRONG TRAINING (64x loop, clean, P+P)"
echo "   template : $TEMPLATE"
echo "   config   : $CONFIG   (executor_mode=P+P, no shared memory)"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)"
echo "=================================================================="

echo "[mod] loaded executor: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null || echo none)  (want clean 1CD139EF)"
[ -e /sys/rvzr_executor ] || { echo "[mod] /sys/rvzr_executor missing — load the module first"; exit 2; }

SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo n/a)
echo "[env] SMT=$SMT (want off)"
[ "$SMT" = on ] && echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null 2>&1 && echo "[env]  -> disabled"
[ "$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null)" = "1" ] && echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog >/dev/null 2>&1

mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S))"
cd "$REPO"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT"
RC=$?
echo "=================================================================="
shopt -s nullglob
V=( "$OUT"/violation-* ); B=( "$OUT"/bugs/violation-* )
echo " violations: ${#V[@]}   arch-mismatch bugs: ${#B[@]}   (in $OUT)"
if [ "${#V[@]}" -gt 0 ]; then
    echo " => LEAK with strong training. Confirm it's cross-domain: the htrace must track the VICTIM's"
    echo "    (vmvictim) input, not vm's own. Latest: $(ls -dt "$OUT"/violation-* | head -1)"
else
    echo " => still frozen at N=$N: stronger BTB training did NOT make the clean injection land."
fi
echo "=================================================================="
exit "$RC"
