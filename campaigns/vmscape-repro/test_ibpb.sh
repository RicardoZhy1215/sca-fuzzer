#!/usr/bin/env bash
#
# test_ibpb.sh — does IBPB-on-VM-switch close the cross-VM BTB residue?
#
# Replays the STRONG TC-1113 residue violation N times and counts reproductions:
#   CLEAN build (srcversion 1CD139EF): expect N/N   (BTB channel OPEN — baseline, we saw 3/3)
#   IBPB  build (srcversion A07CAEF9): expect 0/N   (IBPB flushes BTB on guest entry -> CLOSED)
#
# The mitigation lives in body_macro_switch_h2g (macros.c): `mov ecx,0x49; mov eax,1; wrmsr` (IBPB)
# right before `vmrun`, so the observer enters with a flushed predictor and can't inherit the
# victim's trained BTB entry.
#
# Load the IBPB build first (needs sudo):
#   sudo rmmod rvzr_executor
#   sudo insmod /home/mluo/sca-fuzzer/rvzr/executor_km/rvzr_executor.ko
#   cat /sys/module/rvzr_executor/srcversion    # should read A07CAEF9...
#
# Usage:  bash test_ibpb.sh            # N=5 replays
#         N=10 bash test_ibpb.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-residue/violation-260814-035332}
CONFIG=$V/reproduce.yaml
N=${N:-5}

[ -f "$V/program.asm" ] || { echo "no program.asm in $V"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
SV=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)

echo "=================================================================="
echo " IBPB mitigation test — replay strong TC-1113 residue x$N"
echo "   loaded module : $SV   (clean=1CD139EF open, IBPB=A07CAEF9 expect closed)"
echo "   violation     : $V"
echo "=================================================================="

hits=0
for i in $(seq 1 "$N"); do
    "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$V/program.asm" -i "$V"/input_*.bin \
        >/dev/null 2>&1
    rc=$?
    if [ "$rc" -eq 1 ]; then hits=$((hits+1)); echo "  run $i: exit=1  VIOLATION (residue fired)";
    elif [ "$rc" -eq 0 ]; then echo "  run $i: exit=0  clean";
    else echo "  run $i: exit=$rc ERROR"; fi
done

echo "=================================================================="
echo " RESULT: $hits/$N reproduced   (module $SV)"
if [ "$hits" -eq 0 ]; then
    echo " => residue does NOT reproduce. If this is the IBPB build (A07CAEF9), IBPB CLOSED the channel."
    echo "    (sanity: reload the clean 1CD139EF build and re-run — it should go back to $N/$N.)"
else
    echo " => residue still fires ($hits/$N). If this is the IBPB build, IBPB did NOT close it — investigate."
fi
echo "=================================================================="
