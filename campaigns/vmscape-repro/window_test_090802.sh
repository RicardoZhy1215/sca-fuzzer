#!/usr/bin/env bash
#
# window_test_090802.sh — is the 090802 "leak" the victim's ARCHITECTURAL footprint (measurement
# window spans the victim), or a genuine TRANSIENT channel inside the observer?
#
# Builds a variant of program.min.asm with ONE change: measurement_start is moved from vm_0 (BEFORE
# the victim) to the top of vm_1 (AFTER the victim). So the measurement window no longer spans the
# victim's execution; it covers only the observer's own gadget (faulty loads + retrieval).
#
#   CONTROL  = program.min.asm            (window spans victim)   -> expect ~10/10
#   WINDOW   = program.min.windowtest.asm (window after victim)   -> the test
#
# Interpretation:
#   control 10/10 + window 0/10  -> CONFIRMED architectural artifact: the observer was directly
#                                   measuring the victim's secret-dependent write, not stealing it
#                                   transiently. Not a CPU transient-execution vulnerability.
#   control 10/10 + window 10/10 -> a real transient channel survives with the victim OUT of the
#                                   window -> genuinely speculative; keep investigating.
#
# Usage:
#   bash window_test_090802.sh              # N=10 each
#   N=20 bash window_test_090802.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/reproduce.yaml
CTRL=$V/program.min.asm
WT=$V/program.min.windowtest.asm
N=${N:-10}

[ -f "$CTRL" ] || { echo "no minimized program at $CTRL — run minimize_090802.sh first"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# --- build the window-test variant: drop measurement_start in vm_0, re-insert it at the top of vm_1 ---
awk '
  /\.macro\.measurement_start:/ { next }                                  # remove original (vm_0) start
  { print }
  /\.macro\.landing_h2g\.vm_1:/ { print ".macro.measurement_start: nop qword ptr [rax + 0xff]" }  # re-insert after victim
' "$CTRL" > "$WT"

# sanity: exactly one measurement_start, and it now sits inside vm_1 (after the victim section)
ms=$(grep -c '\.macro\.measurement_start:' "$WT")
[ "$ms" -eq 1 ] || { echo "transform failed: $ms measurement_start lines in $WT"; exit 2; }
echo "=================================================================="
echo " WINDOW TEST — 090802: move measurement_start to AFTER the victim"
echo "   control : $CTRL"
echo "   variant : $WT"
echo "   config  : $CONFIG      inputs: $(ls "$V"/input_*.bin | wc -l) saved      module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "------------------------------------------------------------------"
echo " placement check (measurement_start should be in vm_1, victim writes are in vmvictim_0 above it):"
awk '/\.function_vmvictim_0:/{v=1} /\.function_vm_0:/{v=0}
     /\.section \.data\.vm$/{o=1}
     /measurement_start|measurement_end|function_vm_[012]:|function_vmvictim_0:/{
        tag = (v?"[victim] ":"[obsrvr] "); print "   " tag $0 }' "$WT" | head -20
echo "=================================================================="
cd "$REPO"

reproduce_n () {   # $1 program, $2 label ; sets HITS
    local prog=$1 label=$2 hits=0 rc b a
    echo "--- $label : reproduce x$N  ($(basename "$prog")) ---"
    for i in $(seq 1 "$N"); do
        b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$prog" -i "$V"/input_*.bin >/dev/null 2>&1
        rc=$?
        a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        if [ "$a" -gt "$b" ]; then echo "  !! hpa_gpa EPT COLLISION — reload: bash $CAMP/reload_until_ok.sh"; exit 3; fi
        [ "$rc" -eq 1 ] && hits=$((hits+1))
        printf "  run %2d: exit=%d  %s\n" "$i" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    done
    HITS=$hits; echo "  => $label reproduced $hits/$N"; echo ""
}

reproduce_n "$CTRL" "CONTROL (window spans victim)"; CTRL_HITS=$HITS
reproduce_n "$WT"   "WINDOW  (start after victim) "; WT_HITS=$HITS

echo "=================================================================="
echo " RESULT:  CONTROL $CTRL_HITS/$N    WINDOW $WT_HITS/$N"
if [ "$CTRL_HITS" -gt 0 ] && [ "$WT_HITS" -eq 0 ]; then
    echo " => CONFIRMED: architectural measurement-window artifact. The observer was directly measuring"
    echo "    the victim's secret-dependent write (window overlapped the victim), NOT stealing it"
    echo "    transiently. This is not a CPU transient-execution vulnerability."
elif [ "$CTRL_HITS" -gt 0 ] && [ "$WT_HITS" -gt 0 ]; then
    echo " => a transient channel SURVIVES with the victim out of the window -> genuinely speculative."
    echo "    Keep investigating (this would be the interesting outcome)."
else
    echo " => CONTROL did not reproduce ($CTRL_HITS/$N) — signal down this session. Confirm first:"
    echo "    N=10 MIN=$CTRL V=$V CONFIG=$CONFIG bash $CAMP/verify_min_090802.sh   then re-run."
fi
echo "=================================================================="
