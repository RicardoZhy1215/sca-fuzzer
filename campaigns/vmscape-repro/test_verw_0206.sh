#!/usr/bin/env bash
#
# test_verw_0206.sh — microcode-gated confirmation that the 090802 leak is TSA-L1 (microtag).
#
# Run this AFTER reverting to microcode 0206 (dis_ucode_ldr). It refuses to run on any other ucode.
# AMD's TSA mitigation = VERW clearing the microtag, a capability the 0208 microcode ADDS. So on 0206
# VERW should NOT clear the microtag:
#
#   0208 (already tested): CONTROL 10/10, VERW 0/10   (VERW clears the microtag)
#   0206 (this test):
#     CONTROL leaks, VERW STILL leaks  -> the fix is the 0208 microtag-clear -> CONFIRMED TSA-L1
#     CONTROL leaks, VERW closes        -> VERW clears it regardless of ucode -> NOT the microtag
#
# Uses hpa_gpa=off (separate per-actor physical -> no EPT collision).
#
# Usage:  bash test_verw_0206.sh            # N=10 each
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
SRC_CONFIG=${SRC_CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$SRC_CONFIG" ] || SRC_CONFIG=$V/reproduce.yaml
CONFIG=$CAMP/config-090802-noalias.yaml
CTRL=$V/program.min.asm
VERW=$V/program.min.verw.asm
N=${N:-10}

# --- gate: must be on 0206 ---
uc=$(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)
echo "  microcode: $uc  (want 0xa705206)"
if [ "$uc" != "0xa705206" ]; then
    echo "!! not on 0206 — revert first (add dis_ucode_ldr, update-grub, update-initramfs -u, reboot). Aborting."; exit 2
fi
[ -f "$CTRL" ] || { echo "no minimized program at $CTRL"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# no-alias config (separate physical, no collision)
sed 's/^x86_enable_hpa_gpa_collisions:.*/x86_enable_hpa_gpa_collisions: false/' "$SRC_CONFIG" > "$CONFIG"
grep -q "x86_enable_hpa_gpa_collisions: false" "$CONFIG" || { echo "sed failed on hpa_gpa key"; exit 2; }

# VERW variant: clear the microtag at the top of the observer (vm_1), before its faulting load
awk '{print} /\.macro\.landing_h2g\.vm_1:/ { print "verw word ptr [r14]" }' "$CTRL" > "$VERW"
[ "$(grep -c 'verw word ptr' "$VERW")" -eq 1 ] || { echo "verw insert failed"; exit 2; }

echo "=================================================================="
echo " VERW TEST on 0206 — TSA-L1 microcode-gated confirmation"
echo "   config : $CONFIG (hpa_gpa=off)   module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "   SMT: $(cat /sys/devices/system/cpu/smt/control 2>/dev/null) gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)  (perf_env.sh for a clean run)"
echo "=================================================================="
cd "$REPO"
run () {  # $1 prog $2 label -> HITS
    local prog=$1 label=$2 h=0 rc
    echo "--- $label : x$N ($(basename "$prog")) ---"
    for r in $(seq 1 "$N"); do
        "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$prog" -i "$V"/input_*.bin >/dev/null 2>&1
        rc=$?; [ "$rc" -eq 1 ] && h=$((h+1))
        printf "  run %2d: exit=%d  %s\n" "$r" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    done
    HITS=$h; echo "  => $label $h/$N"; echo ""
}
run "$CTRL" "CONTROL (no verw)"; C=$HITS
run "$VERW" "VERW (clear microtag)"; W=$HITS

echo "=================================================================="
echo " RESULT on 0206:  CONTROL $C/$N   VERW $W/$N   (0208 was: CONTROL 10/10, VERW 0/10)"
if [ "$C" -gt 0 ] && [ "$W" -gt 0 ]; then
    echo " => VERW does NOT help on 0206 -> the fix is the 0208 microtag-clear -> CONFIRMED TSA-L1."
elif [ "$C" -gt 0 ] && [ "$W" -eq 0 ]; then
    echo " => VERW closes it even on 0206 -> clears it regardless of ucode -> NOT the 0208 microtag."
else
    echo " => CONTROL didn't reproduce ($C/$N) — signal down; run perf_env.sh / raise N and retry."
fi
echo "=================================================================="