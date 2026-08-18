#!/usr/bin/env bash
#
# replay_fuzz_u2u.sh — replay the guest-USER -> guest-USER fuzzing violation (TC 19779).
# Same generated program as the u2k case, but vmvictim is guest USER: a guest-user observer leaking
# another guest-user (and/or main) secret via the fault-driven (L1TF/Foreshadow-style) channel.
# NOTE: this leak is layout-fragile (the user victim's "secret injection" only imprints through a
# transient fault window), so it may read 0/N unless the current physical layout happens to reproduce
# it — unlike the kernel-victim (u2k) version, which is layout-robust.
# Thin wrapper over replay_fuzz.sh; all its options (N=, etc.) apply.
exec env V=/home/mluo/sca-fuzzer/campaigns/vmscape-repro/out-fuzz-u2u/violation-260814-125938 \
     CONFIG=/home/mluo/sca-fuzzer/campaigns/vmscape-repro/config-fuzz-u2u-strong.yaml \
     bash /home/mluo/sca-fuzzer/campaigns/vmscape-repro/replay_fuzz.sh "$@"
