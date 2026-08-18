#!/usr/bin/env bash
#
# replay_fuzz_u2k.sh — replay the guest-USER -> guest-KERNEL fuzzing violation (TC 19779).
# Same generated program as the u2u case, but vmvictim is guest KERNEL: a guest-user observer
# leaking guest-kernel (and/or main) data via the fault-driven (L1TF/Foreshadow-style) channel.
# Thin wrapper over replay_fuzz.sh; all its options (N=, etc.) apply.
exec env V=/home/mluo/sca-fuzzer/campaigns/vmscape-repro/out-fuzz-u2k/violation-260814-201415 \
     CONFIG=/home/mluo/sca-fuzzer/campaigns/vmscape-repro/config-fuzz-u2k-strong.yaml \
     bash /home/mluo/sca-fuzzer/campaigns/vmscape-repro/replay_fuzz.sh "$@"
