#!/usr/bin/env bash
#
# run_gs2hu.sh — vBTI_GS->HU: guest-kernel attacker injects, host-user victim mispredicts & leaks.
#
# Requires the executor module built from THIS tree (it carries the alias_target extension that
# lets the guest attacker share an F+R channel with the host-user victim). The script checks the
# loaded module's srcversion against the built .ko and swaps it if needed (sudo).
#
# Usage:
#   bash run_gs2hu.sh              # full campaign (N=10000, I=50)
#   SMOKE=1 bash run_gs2hu.sh      # 5-TC pipeline check
#   N=2000 I=50 bash run_gs2hu.sh  # shorter
#
set -uo pipefail

REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
KO=$REPO/rvzr/executor_km/rvzr_executor.ko
CONFIG=$CAMP/config-gs2hu.yaml
MODE=${1:-aligned}
case "$MODE" in
  # transient-only redesign (recommended): leaks only via mispredicted disclosure
  aligned)          TEMPLATE=$CAMP/template-gs2hu-transient-aligned.asm;        OUT=$CAMP/out-gs2hu-aligned ;;
  aligned-no-jmp)   TEMPLATE=$CAMP/template-gs2hu-transient-aligned-no-jmp.asm; OUT=$CAMP/out-gs2hu-aligned-no-jmp ;;
  transient)        TEMPLATE=$CAMP/template-gs2hu-transient.asm;        OUT=$CAMP/out-gs2hu-transient ;;
  transient-no-jmp) TEMPLATE=$CAMP/template-gs2hu-transient-no-jmp.asm; OUT=$CAMP/out-gs2hu-transient-no-jmp ;;
  # original direct-sharing version (no-jmp also leaks -> not vBTI)
  baseline)         TEMPLATE=$CAMP/template-gs2hu.asm;                  OUT=$CAMP/out-gs2hu ;;
  no-jmp)           TEMPLATE=$CAMP/template-gs2hu-no-jmp.asm;           OUT=$CAMP/out-gs2hu-no-jmp ;;
  *) echo "unknown mode $MODE (transient|transient-no-jmp|baseline|no-jmp)"; exit 2 ;;
esac

N=${N:-10000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=5; I=10; }

# derive the wanted srcversion from the built .ko so a rebuild never desyncs this script
WANT_SRCVER=$(modinfo -F srcversion "$KO" 2>/dev/null || echo unknown)

echo "=================================================================="
echo " vBTI_GS->HU  (guest-kernel attacker -> host-user victim)"
echo "   template : $TEMPLATE"
echo "   config   : $CONFIG   (attacker.data_alias = victim)"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)"
echo "=================================================================="

# --- low-noise env ---
SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo n/a)
echo "[env] SMT=$SMT (want off)"
[ "$SMT" = on ] && echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null 2>&1 && echo "[env]  -> disabled"
[ "$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null)" = "1" ] && echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog >/dev/null 2>&1

# --- ensure THIS tree's module (with alias_target) is loaded ---
[ -f "$KO" ] || { echo "[mod] $KO missing — run: make -C rvzr/executor_km"; exit 2; }
LOADED=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null || echo none)
echo "[mod] loaded=$LOADED want=$WANT_SRCVER"
if [ "$LOADED" != "$WANT_SRCVER" ]; then
    echo "[mod] swapping to the freshly built module (sudo)"
    lsmod | grep -qw rvzr_executor && sudo rmmod rvzr_executor
    sudo insmod "$KO" || { echo "[mod] insmod FAILED"; exit 2; }
    echo "[mod] now loaded: $(cat /sys/module/rvzr_executor/srcversion)"
fi
[ -e /sys/rvzr_executor ] || { echo "[mod] /sys/rvzr_executor missing"; exit 2; }

# --- run ---
mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S))"
cd "$REPO"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT"
RC=$?

echo "=================================================================="
[ "$RC" -ne 0 ] && echo " tfuzz exit code: $RC"
shopt -s nullglob
V=( "$OUT"/violation-* ); B=( "$OUT"/bugs/violation-* )
echo " violations: ${#V[@]}    arch-mismatch bugs: ${#B[@]}    (in $OUT)"
[ "${#V[@]}" -gt 0 ] && echo " => GS->HU leak detected. Latest: $(ls -dt "$OUT"/violation-* | head -1)"
[ "${#V[@]}" -eq 0 ] && echo " => no violation this run"
echo "=================================================================="
exit "$RC"
