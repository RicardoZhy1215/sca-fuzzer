#!/usr/bin/env bash
#
# replay_residue.sh — replay the SAVED cross-VM BTB residue violation directly.
#
# The original residue run detected the violation at TEST CASE 585 and saved everything needed to
# reproduce it: the exact program (program.asm), its 100 inputs, and the config. This replays that one
# test case via `revizor.py reproduce` — seconds instead of re-fuzzing 585 test cases (~8+ min).
#
# Use this to CONFIRM the residue reproduces on the current box/build without the full campaign.
# For the from-scratch fuzzing reproduction, use run_residue.sh instead.
#
# Loads nothing — needs the clean module (1CD139EF) loaded first:
#   sudo insmod /home/mluo/sca-fuzzer/rvzr/executor_km/rvzr_executor.ko
# and the low-latency env pinned (run perf_env.sh once after a reboot) so it isn't noise-throttled.
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-residue/violation-260814-035332}   # strong TC-1113 residue (reproduces 3/3); override V=<dir> for another

[ -f "$V/program.asm" ] || { echo "no program.asm in $V"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "[mod] module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

echo "=================================================================="
echo " REPLAY saved residue violation (TC 585)"
echo "   violation dir : $V"
echo "   program       : $V/program.asm"
echo "   inputs        : $(ls "$V"/input_*.bin | wc -l) saved inputs"
echo "   module        : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want 1CD139EF)"
echo "=================================================================="

cd "$REPO"
"$PY" revizor.py reproduce \
    -s "$BASE" \
    -c "$V/reproduce.yaml" -I "$V" \
    -t "$V/program.asm" \
    -i "$V"/input_*.bin
RC=$?
echo "=================================================================="
echo " reproduce exit=$RC   (revizor exit code = 'violations>0': 1=violation, 0=clean)"
if [ "$RC" -eq 1 ]; then
    echo "   => VIOLATION REPRODUCED: the residue is present on this box/build."
elif [ "$RC" -eq 0 ]; then
    echo "   => NO violation: the saved residue did NOT reproduce in this single replay."
    echo "      Weak signal — before concluding the channel is closed, make sure perf_env.sh was"
    echo "      applied (noise suppresses it) and note a single reproduce pass is less reliable than"
    echo "      the full deterministic run (run_residue.sh)."
else
    echo "   => run error (exit $RC) — not a clean/violation result. Check the log above."
fi
echo "=================================================================="
exit "$RC"
