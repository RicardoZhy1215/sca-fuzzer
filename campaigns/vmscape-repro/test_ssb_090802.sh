#!/usr/bin/env bash
#
# test_ssb_090802.sh — is the 090802 value-forward a Speculative Store Bypass (SSB)?
#
# Run this AFTER rebooting with `spec_store_bypass_disable=on` (SSBD forced globally). It first
# verifies SSBD is actually active, then reproduces the minimized gadget N times.
#
#   baseline (SSBD off/prctl, ucode 0208): 090802 reproduced ~5/5..10/10.
#   now (SSBD forced on):
#     0/N   -> the value-forward WAS store->load bypass (SSB). Mechanism identified.
#     ~N/N  -> NOT SSB. Store-buffer candidate is out too -> fault-specific forward is next.
#
# Usage:
#   bash test_ssb_090802.sh            # N=10 on program.min.asm
#   N=20 bash test_ssb_090802.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/reproduce.yaml
PROG=${PROG:-$V/program.min.asm}
N=${N:-10}

# --- gate: SSBD must actually be forced on ---
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
echo "  spec_store_bypass: $ssb"
echo "  cmdline flag     : $(grep -o 'spec_store_bypass_disable=on' /proc/cmdline || echo 'NOT on cmdline')"
if echo "$ssb" | grep -qi "via prctl"; then
    echo "!! SSBD is still per-prctl (NOT globally on). The flag didn't take — check /etc/default/grub,"
    echo "   re-run update-grub + reboot. Aborting so we don't misread the result."; exit 2
fi
if ! echo "$ssb" | grep -qi "disabled"; then
    echo "!! spec_store_bypass is not showing a global 'disabled' mitigation. Aborting."; exit 2
fi
echo "  -> SSBD is globally ON. Proceeding."

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

echo "=================================================================="
echo " SSB TEST — 090802 with SSBD forced ON"
echo "   program : $PROG  module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  ucode: $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)"
echo "   SMT: $(cat /sys/devices/system/cpu/smt/control 2>/dev/null)  gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "=================================================================="
cd "$REPO"
hits=0
for r in $(seq 1 "$N"); do
    b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
    "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$PROG" -i "$V"/input_*.bin >/dev/null 2>&1
    rc=$?
    a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
    if [ "$a" -gt "$b" ]; then echo "  !! hpa_gpa EPT COLLISION — bash $CAMP/reload_until_ok.sh, then retry"; exit 3; fi
    [ "$rc" -eq 1 ] && hits=$((hits+1))
    printf "  run %2d: exit=%d  %s\n" "$r" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
done
echo "=================================================================="
echo " RESULT (SSBD on): reproduced $hits/$N   (baseline with SSBD off was ~5/5..10/10)"
if [ "$hits" -eq 0 ]; then
    echo " => leak DIED with SSBD on -> it WAS Speculative Store Bypass (store->load). Mechanism = SSB."
else
    echo " => leak SURVIVES with SSBD on -> NOT SSB. Store-buffer candidate ruled out too;"
    echo "    next hypothesis: a fault-specific stale forward (Meltdown-RW-like). "
fi
echo "=================================================================="