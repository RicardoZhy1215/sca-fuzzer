#!/usr/bin/env bash
#
# run_h2v.sh — fuzz the host-kernel -> guest-kernel (H2V) boundary.
#
#   victim   = main (host kernel), trains a BTB entry in host context
#   observer = vm   (guest kernel), measures via F+R after a host->guest switch
#
# Modes:
#   baseline   template-h2v.asm         both sides do an indirect jmp
#   no-jmp     template-h2v-no-jmp.asm  control: no indirect jmp (isolates the
#                                        shared-memory F+R channel from BTB residue)
#
# Usage:
#   bash run_h2v.sh                 # baseline campaign (N=10000, I=50)
#   bash run_h2v.sh no-jmp          # control campaign
#   SMOKE=1 bash run_h2v.sh         # 5-TC pipeline check
#   N=2000 I=50 bash run_h2v.sh     # shorter run
#
set -uo pipefail

REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
KO=$REPO/rvzr/executor_km/rvzr_executor.ko
WANT_SRCVER=$(modinfo -F srcversion "$KO" 2>/dev/null || echo unknown)
CONFIG=$CAMP/config-h2v.yaml

MODE=${1:-baseline}
case "$MODE" in
    baseline) TEMPLATE=$CAMP/template-h2v.asm;        OUT=$CAMP/out-h2v ;;
    no-jmp)   TEMPLATE=$CAMP/template-h2v-no-jmp.asm; OUT=$CAMP/out-h2v-no-jmp ;;
    *) echo "unknown mode '$MODE' (use: baseline | no-jmp)"; exit 2 ;;
esac

N=${N:-10000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=5; I=10; }

echo "=================================================================="
echo " H2V campaign  (host-kernel victim -> guest-kernel observer)"
echo "   mode     : $MODE"
echo "   template : $TEMPLATE"
echo "   config   : $CONFIG"
echo "   campaign : N=$N programs x I=$I inputs   (SMOKE=$SMOKE)"
echo "=================================================================="

# --- low-noise environment ---
SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo n/a)
echo "[env] SMT=$SMT (want off)"
[ "$SMT" = on ] && echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null 2>&1 && echo "[env]  -> disabled"
[ "$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null)" = "1" ] && echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog >/dev/null 2>&1

# --- ensure the guest-capable module is loaded (needed: this test has a guest actor) ---
[ -f "$KO" ] || { echo "[mod] $KO missing — build it first (make -C rvzr/executor_km)"; exit 2; }
LOADED=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null || echo none)
if [ "$LOADED" != "$WANT_SRCVER" ]; then
    echo "[mod] loaded=$LOADED want=$WANT_SRCVER -> swapping (sudo)"
    lsmod | grep -qw rvzr_executor && sudo rmmod rvzr_executor
    sudo insmod "$KO" || { echo "[mod] insmod FAILED"; exit 2; }
else
    echo "[mod] correct module already loaded ($LOADED)"
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
if [ "${#V[@]}" -gt 0 ]; then
    echo " => H2V leak detected. Latest: $(ls -dt "$OUT"/violation-* | head -1)"
    [ "$MODE" = baseline ] && echo "    NEXT: run 'bash run_h2v.sh no-jmp' — if that ALSO violates, the"
    [ "$MODE" = baseline ] && echo "          channel is shared-memory F+R (aliasing), not BTB residue."
else
    echo " => no violation this run"
fi
echo "=================================================================="
exit "$RC"
