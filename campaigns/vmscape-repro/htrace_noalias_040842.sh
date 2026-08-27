#!/usr/bin/env bash
#
# htrace_noalias_040842.sh — does violation-260825-040842 (both-kernel) still reproduce with the
# physical aliasing OFF, and what's the htrace difference?
#
# Derives a config with x86_enable_hpa_gpa_collisions:false (separate per-actor physical), reproduces
# the minimized gadget N times, and prints the "Violations detected" HTrace table (the #39-vs-#89
# difference) from a run that fires.
#
#   reproduces with hpa_gpa OFF  -> the leak survives SEPARATE physical -> it uses the VA/microtag
#                                    (false-completion) path, like 090802. Same byte should leak.
#   does NOT reproduce (0/N)      -> 040842 needs the shared physical page -> it's the shared-L1D
#                                    (Foreshadow-style) path only, NOT the microtag.
#
# Usage:
#   bash htrace_noalias_040842.sh              # N=10 on program.min.asm
#   N=20 bash htrace_noalias_040842.sh
#   PROG=$V/program.asm bash htrace_noalias_040842.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$REPO/violation-260825-040842}
SRC_CONFIG=${SRC_CONFIG:-$V/reproduce.yaml}
[ -f "$SRC_CONFIG" ] || SRC_CONFIG=$V/org-config.yaml
NOALIAS_CONFIG=$CAMP/config-040842-noalias.yaml
if [ -z "${PROG:-}" ]; then
    if [ -f "$V/program.min.asm" ]; then PROG=$V/program.min.asm; else PROG=$V/program.asm; fi
fi
N=${N:-10}
LOG=$V/htrace-noalias.log

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# derive the no-aliasing config (flip only hpa_gpa)
sed 's/^x86_enable_hpa_gpa_collisions:.*/x86_enable_hpa_gpa_collisions: false/' "$SRC_CONFIG" > "$NOALIAS_CONFIG"
grep -q "x86_enable_hpa_gpa_collisions: false" "$NOALIAS_CONFIG" || { echo "sed failed on hpa_gpa key"; exit 2; }

echo "=================================================================="
echo " HTRACE (hpa_gpa OFF) — violation-260825-040842  (both guests KERNEL)"
echo "   program : $PROG  ($(grep -vE '^\s*(#|\.|$)' "$PROG"|grep -cvE '^\s*$') instrs)"
echo "   config  : $NOALIAS_CONFIG (x86_enable_hpa_gpa_collisions: false)"
echo "   inputs  : $(ls "$V"/input_*.bin|wc -l)   counterexample: #39 vs #89"
echo "   module  : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)   ucode: $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)"
echo "   SMT: $(cat /sys/devices/system/cpu/smt/control 2>/dev/null) gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)  (perf_env.sh for clean)"
echo "=================================================================="
cd "$REPO"
hits=0
for r in $(seq 1 "$N"); do
    RUNLOG=$V/htrace-noalias-run$r.log                       # per-run log (not overwritten)
    "$PY" revizor.py reproduce -s "$BASE" -c "$NOALIAS_CONFIG" -I "$V" -t "$PROG" -i "$V"/input_*.bin > "$RUNLOG" 2>&1
    rc=$?
    [ "$rc" -eq 1 ] && hits=$((hits+1))
    printf "  run %2d: exit=%d  %s\n" "$r" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    # print the HTrace rows for EVERY violating run (compact: just the ^/. bitmaps + ID counts)
    if [ "$rc" -eq 1 ] && grep -q "Violations detected" "$RUNLOG"; then
        grep -E "ID:|^\^|^\." "$RUNLOG" | grep -E "ID:|[\^.]{20,}" | sed "s/^/       run$r| /"
    fi
done
cp -f "$V/htrace-noalias-run${N}.log" "$LOG" 2>/dev/null || true    # keep the old single-log name too
echo "=================================================================="
echo " RESULT (hpa_gpa OFF): reproduced $hits/$N"
if [ "$hits" -gt 0 ]; then
    echo " => survives SEPARATE physical -> 040842 uses the VA/microtag (false-completion) path,"
    echo "    like 090802. Confirm the leaked byte is still vmvictim/0xb08 via leaked_values_040842.sh."
else
    echo " => did NOT reproduce with aliasing off -> 040842 needs the shared physical page"
    echo "    (shared-L1D / Foreshadow-style), NOT the microtag. (If signal is weak: perf_env + raise N.)"
fi
echo " full log: $LOG"
echo "=================================================================="