#!/usr/bin/env bash
#
# run_residue.sh — reproduce the ORIGINAL cross-VM BTB residue finding.
#
# The very first vmscape-style result: vmvictim (guest user) trains a BTB entry with an indirect
# jmp; vm (guest user, observer) does its own indirect jmp at the SAME code offset. With no BTB
# scrub across VM-exit/-enter, the observer's jmp transiently retargets to the victim-trained gadget,
# touching its cache lines, which the F+R channel surfaces -> Revizor flags an actor-non-interference
# (ct-ni) violation.
#
#   template : template.asm       (vmvictim trains, vm observes; random_instructions fill the gadgets)
#   config   : config.yaml        (F+R + x86_enable_hpa_gpa_collisions, ct-ni, analyser_stat 0.2)
#
# Fixed seeds => deterministic test-case sequence. The original violation hit at TEST CASE 585,
# so any N past ~586 should catch it (default N=700, ~8-10 min). This is a WEAK signal (BPU residue),
# so a run may need to reach the full sample sizes before it trips.
#
# This script does NOT load/unload the kernel module — it only checks. Load the clean build first:
#   sudo insmod /home/mluo/sca-fuzzer/rvzr/executor_km/rvzr_executor.ko
#   (want srcversion 1CD139EF... = clean: no EPT alias, no host alias, no IBPB)
#
# Usage:
#   bash run_residue.sh            # N=700 x I=50  (deterministic; catches the TC-585 residue)
#   N=2000 bash run_residue.sh     # longer, more chances for the weak signal
#   SMOKE=1 bash run_residue.sh    # 5-TC pipeline check (won't reach 585)
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
CONFIG=$CAMP/config.yaml
TEMPLATE=$CAMP/template.asm
OUT=$CAMP/out-residue
N=${N:-700}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=5; I=10; }

echo "=================================================================="
echo " CROSS-VM BTB RESIDUE  (vmvictim trains -> vm observes, F+R)"
echo "   template : $TEMPLATE"
echo "   config   : $CONFIG   (F+R + hpa_gpa, ct-ni)"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)   [original residue @ TC 585]"
echo "=================================================================="

# --- module check only (no load/unload) ---
if [ ! -e /sys/rvzr_executor ]; then
    echo "[mod] /sys/rvzr_executor MISSING — the executor module is not loaded."
    echo "[mod] load the clean build first, then re-run this script:"
    echo "        sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"
    exit 2
fi
echo "[mod] loaded srcversion: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"

# --- environment (want SMT off, nmi_watchdog off) ---
SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo n/a)
echo "[env] SMT=$SMT (want off)"
[ "$SMT" = on ] && echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null 2>&1 && echo "[env]  -> disabled"
[ "$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null)" = "1" ] && echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog >/dev/null 2>&1

mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S)) — live log below:"
echo "------------------------------------------------------------------"
cd "$REPO"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT"
RC=$?
echo "------------------------------------------------------------------"
shopt -s nullglob
V=( "$OUT"/violation-* )
echo " violations: ${#V[@]}   (in $OUT)"
if [ "${#V[@]}" -gt 0 ]; then
    echo " => CROSS-VM BTB RESIDUE REPRODUCED. Latest: $(ls -dt "$OUT"/violation-* | head -1)"
    echo "    (confirm it's real residue, not noise: the report's htrace must depend on the VICTIM's"
    echo "     input, and removing the vmvictim trainer jmp must make it disappear — see run_residue_ab.sh)"
else
    echo " => no violation at N=$N. This is a weak signal; try N=2000, and confirm the clean"
    echo "    1CD139EF module is loaded (a stale/modified build can suppress it)."
fi
echo "=================================================================="
exit "$RC"
