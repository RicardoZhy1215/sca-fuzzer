#!/usr/bin/env bash
#
# run_k2u.sh — K2U BTI: host-user attacker trains BTB, host-kernel victim mispredicts & discloses.
# Shared F+R channel via host<->host aliasing (user.data_alias=kvictim; needs the executor build
# that has set_host_data_alias). Both actors are in the host address space, so BTB alignment works.
#
# Usage:
#   bash run_k2u.sh              # aligned baseline (N=10000, I=50)
#   bash run_k2u.sh no-jmp       # control (expect clean)
#   SMOKE=1 bash run_k2u.sh      # 5-TC pipeline check
#
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
KO=$REPO/rvzr/executor_km/rvzr_executor.ko
CONFIG=$CAMP/config-k2u-bti.yaml
MODE=${1:-baseline}
case "$MODE" in
  baseline) TEMPLATE=$CAMP/template-k2u-bti.asm;        OUT=$CAMP/out-k2u ;;
  no-jmp)   TEMPLATE=$CAMP/template-k2u-bti-no-jmp.asm; OUT=$CAMP/out-k2u-no-jmp ;;
  poscontrol) TEMPLATE=$CAMP/template-k2u-poscontrol.asm; OUT=$CAMP/out-k2u-poscontrol ;;
  *) echo "unknown mode $MODE (baseline|no-jmp|poscontrol)"; exit 2 ;;
esac
N=${N:-10000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=5; I=10; }
WANT_SRCVER=$(modinfo -F srcversion "$KO" 2>/dev/null || echo unknown)

echo "=================================================================="
echo " K2U BTI  (host-user attacker -> host-kernel victim)   mode=$MODE"
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
    echo "[mod] swapping to the host-alias build (sudo)"
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
[ "${#V[@]}" -gt 0 ] && echo " => K2U leak. Latest: $(ls -dt "$OUT"/violation-* | head -1)"
[ "${#V[@]}" -gt 0 ] && [ "$MODE" = baseline ] && echo "    NEXT: run 'bash run_k2u.sh no-jmp' — if clean, it's genuine BTI."
echo "=================================================================="
exit "$RC"
