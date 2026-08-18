#!/usr/bin/env bash
#
# reload_until_leak_u2k.sh — recover a physical layout that reproduces the guest-USER -> guest-KERNEL
# fuzzing violation (TC 19779, vmvictim=kernel). Same hpa_gpa/L1TF layout-dependence as u2u, so we
# reload the module until a layout both loads and leaks.
#
# Uses the STRONG config (u2k is only ~1/10 at default sampling; strong = ~10/10 on a leaking layout,
# so the search reliably detects a good layout in a couple of tries). Thin wrapper over
# reload_until_leak.sh — all its options (MAX=, REPRO_TRIES=) apply.
#
# ==================================================================================================
#  RUN ONLY ONE EXECUTOR SEARCH AT A TIME. If reload_until_leak.sh (u2u) is still running, STOP it
#  first (Ctrl-C / kill). Two processes reloading + hitting /sys/rvzr_executor at once WILL wedge it.
# ==================================================================================================
#
# Usage:
#   bash reload_until_leak_u2k.sh                # up to 40 reloads
#   MAX=80 bash reload_until_leak_u2k.sh
exec env \
    V=/home/mluo/sca-fuzzer/campaigns/vmscape-repro/out-fuzz-u2k/violation-260814-201415 \
    CONFIG=/home/mluo/sca-fuzzer/campaigns/vmscape-repro/config-fuzz-u2k-strong.yaml \
    REPRO_TRIES="${REPRO_TRIES:-2}" \
    bash /home/mluo/sca-fuzzer/campaigns/vmscape-repro/reload_until_leak.sh "$@"
