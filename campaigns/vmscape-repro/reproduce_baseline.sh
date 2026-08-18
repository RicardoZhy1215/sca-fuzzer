#!/usr/bin/env bash
#
# reproduce_baseline.sh — reproduce the canonical VMSCAPE template on this box.
#
# The "classic" template is campaigns/vmscape-repro/template.asm: a three-actor
# cross-VM indirect-branch-predictor (BTB) residue test.
#
#   main      (host  kernel)            orchestrates VM-enter / VM-exit
#   vmvictim  (guest kernel)            trains a BTB entry via `lea r,[rip+L]; jmp r`
#   vm        (guest kernel, observer)  does its own indirect jmp; if BTB residue
#                                       survives the VM-exit/-enter it transiently
#                                       lands on the victim's gadget, and F+R sees it
#
# A "violation" under the ct-ni contract == the cross-VM BTB channel is live.
# Per the campaign README the first violation typically appears near TC ~1100
# (~16 min) on Zen4 with SMT off and microcode 0x0a705206.
#
# This runs against the sca-fuzzer build (the guest-user-capable tree). It will
# hot-swap the loaded executor module to this tree's build if needed (sudo).
#
# Usage:
#   bash reproduce_baseline.sh              # full campaign (N=10000, I=50)
#   SMOKE=1 bash reproduce_baseline.sh      # 5-TC pipeline check, no real search
#   N=2000 I=50 bash reproduce_baseline.sh  # shorter campaign
#
set -uo pipefail

# --- paths ------------------------------------------------------------------
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
KO=$REPO/rvzr/executor_km/rvzr_executor.ko
WANT_SRCVER=$(modinfo -F srcversion "$KO" 2>/dev/null || echo unknown)

CONFIG=$CAMP/config.yaml
TEMPLATE=$CAMP/template.asm
OUT=$CAMP/out

# --- tunables (env-overridable) ---------------------------------------------
N=${N:-10000}      # number of test-case programs
I=${I:-50}         # inputs per test case
SMOKE=${SMOKE:-0}  # SMOKE=1 -> quick pipeline check (5 TC x 10 inputs)
[ "$SMOKE" = "1" ] && { N=5; I=10; }

echo "=================================================================="
echo " VMSCAPE baseline reproduction  (cross-VM BTB residue)"
echo "   template : $TEMPLATE"
echo "   config   : $CONFIG"
echo "   campaign : N=$N programs x I=$I inputs   (SMOKE=$SMOKE)"
echo "=================================================================="

# --- 1. sanity + low-noise environment --------------------------------------
VENDOR=$(grep -m1 vendor_id /proc/cpuinfo | awk '{print $3}')
UCODE=$(grep -m1 microcode /proc/cpuinfo | awk '{print $3}')
echo "[env] vendor=$VENDOR  microcode=$UCODE  (VMSCAPE is an AMD Zen4 effect)"

SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo unknown)
if [ "$SMT" = "on" ]; then
    echo "[env] SMT is on -> disabling (README: SMT-off is the validated config)"
    echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null || {
        echo "[env] WARNING: could not disable SMT (expect more false positives)"; }
fi
if [ "$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null)" = "1" ]; then
    echo "[env] disabling NMI watchdog to cut measurement noise"
    echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog >/dev/null || true
fi

# --- 2. build the executor module if it is missing --------------------------
if [ ! -f "$KO" ]; then
    echo "[mod] $KO missing -> building"
    make -C "$REPO/rvzr/executor_km" || { echo "[mod] build FAILED"; exit 2; }
fi

# --- 3. make sure THIS tree's module is the loaded one ----------------------
LOADED=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null || echo none)
if [ "$LOADED" != "$WANT_SRCVER" ]; then
    echo "[mod] loaded=$LOADED  want=$WANT_SRCVER  -> swapping (sudo)"
    lsmod | grep -qw rvzr_executor && sudo rmmod rvzr_executor
    sudo insmod "$KO" || { echo "[mod] insmod FAILED — is it built for $(uname -r)?"; exit 2; }
else
    echo "[mod] correct module already loaded ($LOADED)"
fi
[ -e /sys/rvzr_executor ] || { echo "[mod] /sys/rvzr_executor missing — module did not init"; exit 2; }

# --- 4. run the campaign ----------------------------------------------------
mkdir -p "$OUT"
echo "[run] tfuzz starting  ($(date +%H:%M:%S))"
cd "$REPO"
"$PY" revizor.py tfuzz \
    -s "$BASE" \
    -c "$CONFIG" \
    -t "$TEMPLATE" \
    -n "$N" -i "$I" \
    -w "$OUT"
RC=$?

# --- 5. verdict -------------------------------------------------------------
echo "=================================================================="
[ "$RC" -ne 0 ] && echo " tfuzz exit code: $RC"
shopt -s nullglob
VDIRS=( "$OUT"/violation-* )
echo " violation dirs in $OUT : ${#VDIRS[@]}"
if [ "${#VDIRS[@]}" -gt 0 ]; then
    LATEST=$(ls -dt "$OUT"/violation-* | head -1)
    echo " => cross-VM BTB residue REPRODUCED. Latest: $LATEST"
    echo "    inspect: $LATEST/program.asm , report.txt"
    echo "    minimize: $PY $REPO/revizor.py minimize -s $BASE -c $LATEST/reproduce.yaml \\"
    echo "              -t $LATEST/program.asm -i $LATEST/input_*.bin --num-inputs $I"
else
    echo " => no violation this run"
    [ "$SMOKE" = "1" ] && echo "    (SMOKE run only checks the pipeline; it does not search)"
fi
echo "=================================================================="
exit "$RC"
