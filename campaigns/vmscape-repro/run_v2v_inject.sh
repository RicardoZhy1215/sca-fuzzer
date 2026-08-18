#!/usr/bin/env bash
#
# run_v2v_inject.sh — V2V INJECTION: guest-user attacker trains the BTB, guest-user victim
# mispredicts into a disclosure gadget and leaks its own secret. Both actors are guests, so their
# indirect jmps share a virtual address (BTB alignment is free) — this isolates "does injection work
# at all?" from the alignment wall that blocks K2U/GS->HU. Shared F+R channel via
# x86_enable_hpa_gpa_collisions (same as the residue test).
#
# Usage:
#   bash run_v2v_inject.sh            # injection (N=10000, I=50)
#   bash run_v2v_inject.sh no-jmp     # control (expect clean)
#   SMOKE=1 bash run_v2v_inject.sh    # 5-TC pipeline check
#
# Requires the EPT-physical-alias executor build (srcversion 81507044...): the observer's MAIN page
# is EPT-backed by the victim's frame so the two guests share a real F+R channel. The script hot-swaps
# to whatever .ko is built, so just `make -C rvzr/executor_km` first if needed.
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
KO=$REPO/rvzr/executor_km/rvzr_executor.ko
CONFIG=$CAMP/config-v2v-inject.yaml
MODE=${1:-inject}
case "$MODE" in
  inject) TEMPLATE=$CAMP/template-v2v-inject.asm;        OUT=$CAMP/out-v2v-inject ;;
  no-jmp) TEMPLATE=$CAMP/template-v2v-inject-no-jmp.asm; OUT=$CAMP/out-v2v-inject-no-jmp ;;
  *) echo "unknown mode $MODE (inject|no-jmp)"; exit 2 ;;
esac
N=${N:-10000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=5; I=10; }
WANT_SRCVER=$(modinfo -F srcversion "$KO" 2>/dev/null || echo unknown)

echo "=================================================================="
echo " V2V INJECTION  (guest-user attacker -> guest-user victim)   mode=$MODE"
echo "   template : $TEMPLATE"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)"
echo "=================================================================="

SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo n/a)
echo "[env] SMT=$SMT (want off)"
[ "$SMT" = on ] && echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null 2>&1 && echo "[env]  -> disabled"
[ "$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null)" = "1" ] && echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog >/dev/null 2>&1

[ -f "$KO" ] || { echo "[mod] $KO missing"; exit 2; }
LOADED=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null || echo none)
echo "[mod] loaded=$LOADED want=$WANT_SRCVER"
if [ "$LOADED" != "$WANT_SRCVER" ]; then
    echo "[mod] swapping to the built module (sudo)"
    lsmod | grep -qw rvzr_executor && sudo rmmod rvzr_executor
    sudo insmod "$KO" || { echo "[mod] insmod FAILED"; exit 2; }
fi
[ -e /sys/rvzr_executor ] || { echo "[mod] /sys/rvzr_executor missing"; exit 2; }

mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S))"
cd "$REPO"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT"
RC=$?
echo "=================================================================="
shopt -s nullglob
V=( "$OUT"/violation-* ); B=( "$OUT"/bugs/violation-* )
echo " violations: ${#V[@]}   arch-mismatch bugs: ${#B[@]}   (in $OUT)"
if [ "${#V[@]}" -gt 0 ] && [ "$MODE" = inject ]; then
    echo " => FULL vBTI LANDS (steering + cross-domain readout). Run no-jmp to confirm it's the BTB."
fi
if [ "${#V[@]}" -eq 0 ] && [ "$MODE" = inject ]; then
    echo " => clean, expected with the CURRENT executor: predictor STEERING works (proven by the"
    echo "    residue violation, see FINDINGS-injection.md), but two guests do NOT share a physical"
    echo "    F+R frame, so the observer can't read the victim's imprint. Needs the EPT-alias fix."
fi
echo "=================================================================="
exit "$RC"
