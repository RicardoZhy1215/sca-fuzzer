#!/usr/bin/env bash
#
# run_gk2hk.sh — fuzz the guest-KERNEL (observer/attacker) -> host-KERNEL (victim) direction.
#
#   The HOST (main) holds the secret and runs the fuzzed injection; the GUEST (vm) is the observer.
#   A ct-ni violation = a guest transiently read the hypervisor's data (VM-escape / "vmscape").
#
#   NOTE: this template is NEW (the HOST runs random_instructions as the victim — the other templates
#   only ever fuzz guests). Run SMOKE=1 FIRST to confirm the generator/executor pipeline is happy
#   before a long campaign. Needs the clean module (1CD139EF) + perf_env.sh. Config forces
#   hpa_gpa=true (host<->guest share no VA -> the guest reaches the host L1D only via aliasing).
#
# Usage:
#   SMOKE=1 bash run_gk2hk.sh          # 10-TC pipeline check — DO THIS FIRST
#   bash run_gk2hk.sh                  # N=50000 x I=50, random seed
#   N=100000 bash run_gk2hk.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
HERE=$CAMP/gk2hk
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
TEMPLATE=$HERE/template-gk2hk.asm
CONFIG=$HERE/config-gk2hk.yaml
N=${N:-50000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=10; I=10; }
OUT=$HERE/out-gk2hk
NONSTOP=${NONSTOP:-1}; NS_FLAG=(); [ "$NONSTOP" = "1" ] && NS_FLAG=(--nonstop)

echo "=================================================================="
echo " FUZZ guest-KERNEL (observer) -> host-KERNEL (victim)   [VM-escape direction]"
echo "   template : $TEMPLATE   (HOST does the injection; GUEST vm is the observer)"
echo "   config   : $CONFIG     (hpa_gpa=true, ct-ni, threshold 0.5, seed=0 -> random)"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)"
echo "=================================================================="
[ -f "$TEMPLATE" ] || { echo "no template at $TEMPLATE"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "[mod] module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
echo "[mod] loaded: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
echo "[env] governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)  (want performance/off; run perf_env.sh)"
echo "[ucode] $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)"

mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S)) — live log:"
echo "------------------------------------------------------------------"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT" "${NS_FLAG[@]}"
RC=$?
echo "------------------------------------------------------------------"
shopt -s nullglob
Vs=( "$OUT"/violation-* )
echo " violations: ${#Vs[@]}   (in $OUT)"
if [ "${#Vs[@]}" -gt 0 ]; then
    echo " => candidate guest->host leak(s). Latest: $(ls -dt "$OUT"/violation-* | head -1)"
    echo "    triage: confirm the observer(vm)'s input is byte-identical across the counterexample,"
    echo "    and that the leaked address decodes to actor 0 (main/HOST). Minimize + leaked-values next."
else
    echo " => no violation at N=$N. Raise N or re-run (random seed each time). If EVERY TC errors,"
    echo "    suspect the host-injection template — re-run SMOKE=1 and read the pipeline log."
fi
echo "=================================================================="
exit "$RC"
