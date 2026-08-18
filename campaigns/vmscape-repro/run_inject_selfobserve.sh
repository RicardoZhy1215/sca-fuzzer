#!/usr/bin/env bash
#
# run_inject_selfobserve.sh — INJECTION STEERING (self-observe): attacker (vm) trains the BTB,
# VICTIM (vmvictim) mispredicts into a disclosure and leaks ITS OWN secret, and the victim reads
# its own footprint via F+R (readout kept inside the mispredictor -> no cross-actor P+P wall).
#
# This is the residue mechanism with roles swapped (vm=trainer, vmvictim=observer). It proves the
# attacker STEERS the victim's branch — the real injection primitive — with NO EPT alias. It does
# not claim attacker-side exfiltration (that needs L2/LLC P+P).
#
#   template : template-inject-selfobserve.asm
#   config   : config-inject-selfobserve.yaml   (F+R + hpa_gpa; observer = vmvictim)
#
# Deterministic seeds; expect a hit at roughly the same rarity as residue (~1/hundreds..2000 TCs).
# Needs the clean module (1CD139EF) + perf_env.sh pinned. Does NOT touch the module.
#
# CONFIRM any violation is genuine steering (not a random_instructions transient):
#   V=<violation-dir> bash reproduce_trainer_test.sh   # strips the attacker 'jmp rax' -> must go clean
#
# Usage:
#   bash run_inject_selfobserve.sh          # N=2000 x I=50
#   N=700 bash run_inject_selfobserve.sh
#   SMOKE=1 bash run_inject_selfobserve.sh  # 5-TC pipeline check
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
CONFIG=$CAMP/config-inject-selfobserve.yaml
TEMPLATE=$CAMP/template-inject-selfobserve.asm
OUT=$CAMP/out-inject-selfobserve
N=${N:-2000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=5; I=10; }

echo "=================================================================="
echo " INJECTION STEERING (self-observe): vm trains -> vmvictim mispredicts+leaks (F+R self-read)"
echo "   template : $TEMPLATE"
echo "   config   : $CONFIG   (F+R + hpa_gpa; observer=vmvictim)"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)"
echo "=================================================================="

[ -e /sys/rvzr_executor ] || { echo "[mod] module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
echo "[mod] loaded: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
echo "[env] governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)  (want performance/off; run perf_env.sh if not)"

mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S)) — live log:"
echo "------------------------------------------------------------------"
cd "$REPO"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT"
RC=$?
echo "------------------------------------------------------------------"
shopt -s nullglob
V=( "$OUT"/violation-* )
echo " violations: ${#V[@]}   (in $OUT)"
if [ "${#V[@]}" -gt 0 ]; then
    echo " => attacker(vm) STEERS victim(vmvictim) into a transient leak. Latest: $(ls -dt "$OUT"/violation-* | head -1)"
    echo "    CONFIRM: V=<that dir> bash reproduce_trainer_test.sh  (attacker jmp->nop must go clean)"
else
    echo " => clean at N=$N. Weak signal like residue — try N=2000, confirm perf_env.sh + clean module."
fi
echo "=================================================================="
exit "$RC"
