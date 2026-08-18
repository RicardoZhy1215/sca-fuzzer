#!/usr/bin/env bash
#
# reload_until_ok.sh — clear the hpa_gpa EPT-collision that makes every test-case load EIO.
#
# With x86_enable_hpa_gpa_collisions on, guest GPAs are real physical addresses (they change per
# `insmod`). When two land in the same EPT L1 slot the executor asserts in set_extended_page_tables
# -> allocate_sandbox fails -> `OSError [Errno 5]` on /sys/rvzr_executor/test_case. It's a per-load
# roll of the dice, so we just reload the module until a non-colliding layout, then verify a load.
#
# Needs sudo (rmmod/insmod). Run after a reboot, or any time loads start EIOing.
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
KO=$REPO/rvzr/executor_km/rvzr_executor.ko
PY=${PY:-/home/mluo/miniconda3/bin/python3}
V=${V:-$REPO/campaigns/vmscape-repro/out-fuzz-u2u/violation-260814-125938}
MAX=${MAX:-8}

for i in $(seq 1 "$MAX"); do
    echo "=== reload attempt $i/$MAX ==="
    lsmod | grep -qw rvzr_executor && sudo rmmod rvzr_executor
    sudo insmod "$KO" || { echo "insmod failed"; exit 2; }
    # trigger one test-case load and see if the EPT assertion fires
    before=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
    timeout 90 "$PY" "$REPO/revizor.py" reproduce -s "$REPO/base.json" -c "$V/reproduce.yaml" \
        -I "$V" -t "$V/program.asm" -i "$V"/input_*.bin >/dev/null 2>&1 || true
    after=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
    if [ "$after" -gt "$before" ]; then
        echo "  -> EPT collision again (assertion fired). reloading..."
    else
        echo "  -> OK: no EPT collision. executor usable (srcversion $(cat /sys/module/rvzr_executor/srcversion))"
        echo "     now re-apply the low-latency env before measuring:  sudo bash campaigns/vmscape-repro/perf_env.sh"
        exit 0
    fi
done
echo "still colliding after $MAX reloads — reboot for fresh memory, or reduce actor/page count."
exit 1
