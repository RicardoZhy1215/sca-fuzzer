#!/usr/bin/env bash
#
# run_inject_attacker.sh — INJECTION: attacker (vm) trains the BTB, VICTIM (vmvictim) is steered,
# victim's transient leaks ITS OWN secret, attacker reads it via P+P (no shared memory).
#
# Roles are the working residue template swapped (see template-v2v-inject-attacker.asm):
#   vm       = attacker : prime -> train -> probe
#   vmvictim = victim   : loads secret, jmp inherits attacker's BTB entry, mispredicts, leaks
#
# Channel: P+P (config-v2v-pp.yaml) — cross-actor L1D contention, the realistic vmscape channel.
# Needs the clean module (1CD139EF) + perf_env.sh pinned. Does NOT touch the module.
#
# If it VIOLATES, prove it's genuine injection (not a generic transient) with the SAME tools:
#   V=<violation-dir> bash reproduce_trainer_test.sh   # strips the attacker 'jmp rax' -> must go clean
# and a counterexample-swap (only vmvictim's input should drive the htrace).
#
# Usage:
#   bash run_inject_attacker.sh          # N=2000 x I=50
#   N=500 bash run_inject_attacker.sh
#   SMOKE=1 bash run_inject_attacker.sh  # 5-TC pipeline check
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
CONFIG=$CAMP/config-v2v-pp.yaml
TEMPLATE=$CAMP/template-v2v-inject-attacker.asm
OUT=$CAMP/out-inject-attacker
N=${N:-2000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=5; I=10; }

echo "=================================================================="
echo " INJECTION: attacker(vm) trains -> victim(vmvictim) mispredicts+leaks -> attacker probes (P+P)"
echo "   template : $TEMPLATE"
echo "   config   : $CONFIG   (executor_mode=P+P, no shared memory)"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)"
echo "=================================================================="

[ -e /sys/rvzr_executor ] || { echo "[mod] module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
echo "[mod] loaded: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
echo "[env] governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)  (want performance / off; run perf_env.sh if not)"

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
    echo " => candidate cross-domain INJECTION. Latest: $(ls -dt "$OUT"/violation-* | head -1)"
    echo "    CONFIRM: V=<that dir> bash reproduce_trainer_test.sh  (attacker jmp->nop must go clean),"
    echo "    and check the htrace tracks vmvictim's input, not vm's own."
else
    echo " => clean at N=$N. Consistent with the prior finding that the injection direction does not"
    echo "    fire cleanly (only residue does). See the notes in the template header."
fi
echo "=================================================================="
exit "$RC"
