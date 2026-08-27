#!/usr/bin/env bash
#
# test_plainread_vs_verw.sh — is VERW's closing a real CLEAR, or just an incidental memory access?
#
# Inserts, at the top of the observer (vm_1, before its faulting load), one of:
#   CONTROL : nothing
#   VERW    : verw word ptr [r14]        (clears buffers/microtag AND touches [r14])
#   READ    : mov rax, qword ptr [r14]   (just touches [r14], no clear)
# and reproduces each. hpa_gpa=off (no collision).
#
#   READ closes it too (like VERW) -> VERW's effect is the incidental memory access, NOT a clear
#                                     -> the VERW evidence is meaningless for mechanism ID.
#   READ does NOT close, VERW does  -> VERW does a wholesale clear a plain read doesn't
#                                     -> supports a real buffer/microtag clear (still can't tell which).
#
# Usage:  bash test_plainread_vs_verw.sh          # N=10 each
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
READ=$V/program.min.read.asm
N=${N:-10}

[ -f "$CTRL" ] || { echo "no minimized program at $CTRL"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded"; exit 2; }
sed 's/^x86_enable_hpa_gpa_collisions:.*/x86_enable_hpa_gpa_collisions: false/' "$SRC_CONFIG" > "$CONFIG"
awk '{print} /\.macro\.landing_h2g\.vm_1:/ { print "verw word ptr [r14]" }'        "$CTRL" > "$VERW"
awk '{print} /\.macro\.landing_h2g\.vm_1:/ { print "mov rax, qword ptr [r14]" }'   "$CTRL" > "$READ"

echo "=================================================================="
echo " PLAIN-READ vs VERW — is VERW's closing a clear or an incidental memory access?"
echo "   ucode: $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)   module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "   SMT: $(cat /sys/devices/system/cpu/smt/control 2>/dev/null) gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "=================================================================="
cd "$REPO"
run () {  local prog=$1 label=$2 h=0 rc
    echo "--- $label : x$N ($(basename "$prog")) ---"
    for r in $(seq 1 "$N"); do
        "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$prog" -i "$V"/input_*.bin >/dev/null 2>&1
        rc=$?; [ "$rc" -eq 1 ] && h=$((h+1))
        printf "  run %2d: exit=%d  %s\n" "$r" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    done
    HITS=$h; echo "  => $label $h/$N"; echo ""
}
run "$CTRL" "CONTROL (nothing)"; C=$HITS
run "$VERW" "VERW (clear+read)"; W=$HITS
run "$READ" "READ (plain read)"; R=$HITS
echo "=================================================================="
echo " RESULT:  CONTROL $C/$N   VERW $W/$N   READ $R/$N"
if [ "$C" -gt 0 ] && [ "$W" -eq 0 ] && [ "$R" -gt 0 ]; then
    echo " => READ does NOT close but VERW does -> VERW does a real wholesale CLEAR (buffer/microtag),"
    echo "    not just a memory access. Consistent with a data-sampling channel (incl. TSA-L1)."
elif [ "$C" -gt 0 ] && [ "$W" -eq 0 ] && [ "$R" -eq 0 ]; then
    echo " => READ closes it too -> VERW's closing is the INCIDENTAL memory access, not a clear."
    echo "    The VERW evidence tells us nothing about the mechanism."
else
    echo " => inconclusive (CONTROL $C/$N). Run perf_env.sh / raise N and retry."
fi
echo "=================================================================="