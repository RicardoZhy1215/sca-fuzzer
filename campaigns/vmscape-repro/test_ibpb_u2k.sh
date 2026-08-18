#!/usr/bin/env bash
#
# test_ibpb_u2k.sh — does IBPB-on-VM-switch also close the guest-USER -> guest-KERNEL injection?
#
# Replays the confirmed u->k steering violation (attacker vm=guest user trains -> victim
# vmvictim=guest kernel mispredicts+self-leaks) N times and counts reproductions:
#   CLEAN build (srcversion 1CD139EF): expect N/N   (channel OPEN — trainer-strip earlier gave 3/3)
#   IBPB  build (srcversion A07CAEF9): expect 0/N   (IBPB flushes BTB on guest entry -> CLOSED)
#
# Same mitigation as test_ibpb.sh (residue), now checked across the user->kernel privilege boundary.
# Load the build you want to test first (needs sudo):
#   sudo rmmod rvzr_executor; sudo insmod /home/mluo/sca-fuzzer/rvzr/executor_km/rvzr_executor.ko
#   cat /sys/module/rvzr_executor/srcversion    # A07CAEF9 = IBPB, 1CD139EF = clean
#
# Usage:  bash test_ibpb_u2k.sh          # N=5
#         N=10 bash test_ibpb_u2k.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-inject-u2k/violation-260814-064617}
CONFIG=$V/reproduce.yaml
N=${N:-5}

[ -f "$V/program.asm" ] || { echo "no program.asm in $V"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
SV=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)

echo "=================================================================="
echo " IBPB test on guest-USER -> guest-KERNEL injection — replay x$N"
echo "   loaded module : $SV   (clean=1CD139EF open, IBPB=A07CAEF9 expect closed)"
echo "   violation     : $V"
echo "=================================================================="

hits=0
for i in $(seq 1 "$N"); do
    "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$V/program.asm" -i "$V"/input_*.bin \
        >/dev/null 2>&1
    rc=$?
    if [ "$rc" -eq 1 ]; then hits=$((hits+1)); echo "  run $i: exit=1  VIOLATION (u->k steering fired)";
    elif [ "$rc" -eq 0 ]; then echo "  run $i: exit=0  clean";
    else echo "  run $i: exit=$rc ERROR"; fi
done

echo "=================================================================="
echo " RESULT: $hits/$N reproduced   (module $SV)"
if [ "$hits" -eq 0 ]; then
    echo " => u->k injection does NOT reproduce. If this is the IBPB build (A07CAEF9), IBPB CLOSED it too."
    echo "    (sanity: reload clean 1CD139EF and re-run -> should return to $N/$N.)"
else
    echo " => u->k still fires ($hits/$N). If this is the IBPB build, IBPB did NOT close it — investigate."
fi
echo "=================================================================="
