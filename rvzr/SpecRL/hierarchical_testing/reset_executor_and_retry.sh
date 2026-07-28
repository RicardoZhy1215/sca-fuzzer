#!/usr/bin/env bash
# Reset the wedged rvzr_executor kernel module and re-run the reproduce.
#
# Why: `cat /sys/rvzr_executor/trace` was returning EIO. The KM's dbg_dump_mode
# showed n_inputs=500 + quick_and_dirty_mode=1 -- leftover state from a training
# run that was killed mid-measurement. The module rejects our fresh 100-input
# test case, so trace_test_case() fails -> -EIO. The program itself is fully
# sandboxed (all [r14+reg] accesses masked to 13 bits) and never faults, so this
# is NOT a fault-handling / contract problem -- the module just needs a reset.
#
# Needs sudo (rmmod/insmod). Run:  bash reset_executor_and_retry.sh
set -u
KM=/home/hz25d/beyond_v1_v4/sca-fuzzer/rvzr/executor_km
HT=/home/hz25d/beyond_v1_v4/sca-fuzzer/rvzr/SpecRL/hierarchical_testing

echo "== 1. abort if a training run is using the executor =="
if pgrep -af 'train\.py' >/dev/null 2>&1; then
    echo "!! train.py is running -- stop it first (it drives the same KM). Aborting."
    exit 1
fi

echo "== 2. clean up lingering reproduce / timeout wrappers =="
pkill -9 -f 'revizor\.py reproduce'      2>/dev/null || true
pkill -9 -f 'timeout --signal=INT'       2>/dev/null || true
sleep 1

echo "== 3. reload the kernel module =="
if lsmod | grep -q '^rvzr_executor'; then
    sudo rmmod rvzr_executor || { echo "rmmod failed (module busy?)"; exit 1; }
fi
cd "$KM" || { echo "cannot cd $KM"; exit 1; }
make || { echo "!! KM build failed (need kernel headers / Intel host)"; exit 1; }
sudo insmod rvzr_executor.ko || { echo "insmod failed"; exit 1; }
echo "-- KM sysfs present:"; ls /sys/rvzr_executor/ >/dev/null 2>&1 && echo "   OK" || { echo "   MISSING"; exit 1; }
echo "-- KM state after reload (should NOT say n_inputs=500 / quick_and_dirty=1):"
grep -aE 'n_inputs|quick_and_dirty' /sys/rvzr_executor/dbg_dump_mode 2>/dev/null || echo "   (fresh, no test case loaded yet)"

echo "== 4. re-run reproduce =="
bash "$HT/reproduce_check.sh"
