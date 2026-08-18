#!/usr/bin/env bash
#
# run_v2v_fuzz.sh — fuzz the V2V injection. Same attacker-trains / victim-mispredicts structure as
# run_v2v_inject.sh, but the gadget bodies are random_instructions so the fuzzer explores them (the
# clean hand-written version was frozen; the residue leak only fired with fuzzer-filled code).
# Shared cross-domain F+R channel via the EPT alias (config: vm.data_alias = vmvictim).
#
# Usage:
#   bash run_v2v_fuzz.sh            # N=20000 x I=50 (leaks are ~1/hundreds of programs)
#   SMOKE=1 bash run_v2v_fuzz.sh    # 5-TC pipeline check
#
# Needs the EPT-alias executor build (srcversion 81507044...). A violation here is a candidate
# cross-domain vBTI; confirm it's cross-domain by checking it depends on the VICTIM's (vmvictim)
# input, not vm's own — see the note printed at the end.
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
KO=$REPO/rvzr/executor_km/rvzr_executor.ko
CONFIG=$CAMP/config-v2v-inject.yaml
TEMPLATE=$CAMP/template-v2v-inject-fuzz.asm
OUT=$CAMP/out-v2v-fuzz
N=${N:-20000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=5; I=10; }
WANT_SRCVER=$(modinfo -F srcversion "$KO" 2>/dev/null || echo unknown)

echo "=================================================================="
echo " V2V INJECTION (fuzzed)  guest-user attacker -> guest-user victim"
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
    echo "[mod] swapping to the EPT-alias build (sudo)"
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
if [ "${#V[@]}" -gt 0 ]; then
    echo " => candidate vBTI leak(s). Latest: $(ls -dt "$OUT"/violation-* | head -1)"
    echo "    CONFIRM CROSS-DOMAIN: open the report; the leak must vary with the VICTIM's (vmvictim,"
    echo "    actor 1) input. If it only tracks vm's own input, it's plain residue, not injection."
else
    echo " => clean: no leak in the vm->vmvictim direction even fuzzed. Next: try the residue"
    echo "    direction (vmvictim trains, vm mispredicts) with the alias, or widen N."
fi
echo "=================================================================="
exit "$RC"
