#!/usr/bin/env bash
#
# run_v2v_pp.sh — REALISTIC cross-domain vBTI injection: NO shared memory.
#
# Two guest actors, Prime+Probe channel (shared L1D set contention — the actual vmscape exfiltration
# mechanism), and NO data_alias (each guest keeps its own physical frame). The observer trains the
# BTB, the victim mispredicts into its disclosure and transiently touches its own secret, evicting an
# L1D set; the observer probes and recovers the secret. random_instructions fill the gadgets so the
# misprediction fires (a clean hand-written gadget stays frozen).
#
# This is the honest version of the reproduction — no forced actor-to-actor physical sharing.
#
# NOTE: this script does NOT touch the kernel module. Load whichever build you want yourself
# (`sudo rmmod rvzr_executor; sudo insmod rvzr/executor_km/rvzr_executor.ko`) before running.
#
# Usage:
#   bash run_v2v_pp.sh            # N=200 x I=50 (the leak shows up on TC 0)
#   SMOKE=1 bash run_v2v_pp.sh    # tiny pipeline check
#
# A violation here is a candidate cross-domain vBTI. Confirm (as done during development):
#   - same contract hash, different htrace          -> speculative, not architectural
#   - program has indirect jmps and 0 jcc           -> BTB, not Spectre-v1
#   - counterexample inputs: observer (actor 2) data IDENTICAL, only vmvictim (actor 1) differs,
#     and a controlled swap shows the victim (not main) drives the trace -> cross-domain
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
CONFIG=$CAMP/config-v2v-pp.yaml
TEMPLATE=$CAMP/template-v2v-sls-barrier.asm
OUT=$CAMP/out-v2v-sls
N=${N:-200}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=5; I=10; }

echo "=================================================================="
echo " V2V INJECTION-vs-SLS test (SLS barrier, P+P, no shared memory)"
echo "   template : $TEMPLATE"
echo "   config   : $CONFIG   (executor_mode=P+P, no data_alias)"
echo "   campaign : N=$N x I=$I   (SMOKE=$SMOKE)"
echo "=================================================================="

SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo n/a)
echo "[env] SMT=$SMT (want off)"
[ "$SMT" = on ] && echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null 2>&1 && echo "[env]  -> disabled"
[ "$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null)" = "1" ] && echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog >/dev/null 2>&1

mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S))"
cd "$REPO"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT"
RC=$?
echo "=================================================================="
shopt -s nullglob
V=( "$OUT"/violation-* ); B=( "$OUT"/bugs/violation-* )
echo " violations: ${#V[@]}   arch-mismatch bugs: ${#B[@]}   (in $OUT)"
[ "${#V[@]}" -gt 0 ] && echo " => candidate cross-domain vBTI (no shared memory). Latest: $(ls -dt "$OUT"/violation-* | head -1)"
echo "=================================================================="
exit "$RC"
