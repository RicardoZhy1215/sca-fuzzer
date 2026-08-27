#!/usr/bin/env bash
#
# hpa_gpa_test_090802.sh — does the 090802 leak need the hpa_gpa physical aliasing?
#
# x86_enable_hpa_gpa_collisions makes the guests' GPAs alias the host's HPAs (the Foreshadow-VM
# threat model). This turns it OFF and checks whether the violation survives.
#
#   CONTROL = config-refuzz-u2k.yaml       (hpa_gpa = true, as found)   -> expect ~5/5
#   NOALIAS = x86_enable_hpa_gpa_collisions: false                       -> the test
#
# Same program (program.min.asm) + same inputs both times; only the aliasing changes. The setting is
# applied per-run via /sys/rvzr_executor/enable_hpa_gpa_collisions, so no module reload is needed.
#
# Interpretation:
#   control 5/5 + noalias 0/5  -> the cross-actor leak REQUIRES the aliasing (expected — its reach
#                                 depends on the manufactured GPA=HPA collision).
#   control 5/5 + noalias 5/5  -> the leak does NOT need the aliasing -> far more significant
#                                 (cross-actor signal without a manufactured physical collision).
#
# Usage:
#   bash hpa_gpa_test_090802.sh              # N=5 each
#   N=10 bash hpa_gpa_test_090802.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CTRL_CONFIG=${CTRL_CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CTRL_CONFIG" ] || CTRL_CONFIG=$V/reproduce.yaml
NOALIAS_CONFIG=$CAMP/config-090802-noalias.yaml
PROG=${PROG:-$V/program.min.asm}
N=${N:-5}

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# derive the no-aliasing config: flip the single hpa_gpa line to false, change nothing else
sed 's/^x86_enable_hpa_gpa_collisions:.*/x86_enable_hpa_gpa_collisions: false/' "$CTRL_CONFIG" > "$NOALIAS_CONFIG"
grep -q "x86_enable_hpa_gpa_collisions: false" "$NOALIAS_CONFIG" || { echo "sed failed — key not found in $CTRL_CONFIG"; exit 2; }

echo "=================================================================="
echo " HPA_GPA TEST — 090802: does the leak survive with aliasing OFF?"
echo "   program : $PROG  ($(grep -vE '^\s*(#|\.|$)' "$PROG" | grep -cvE '^\s*$') instrs)"
echo "   inputs  : $(ls "$V"/input_*.bin|wc -l)   module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "   microcode: $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)"
echo "   CONTROL : hpa_gpa=true  ($CTRL_CONFIG)"
echo "   NOALIAS : hpa_gpa=false ($NOALIAS_CONFIG)"
echo "=================================================================="
cd "$REPO"

reproduce_n () {   # $1 config, $2 label ; sets HITS
    local cfg=$1 label=$2 hits=0 rc
    echo "--- $label : reproduce x$N ---"
    for r in $(seq 1 "$N"); do
        "$PY" revizor.py reproduce -s "$BASE" -c "$cfg" -I "$V" -t "$PROG" -i "$V"/input_*.bin >/dev/null 2>&1
        rc=$?
        [ "$rc" -eq 1 ] && hits=$((hits+1))
        printf "  run %d: exit=%d  %s\n" "$r" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    done
    HITS=$hits; echo "  => $label reproduced $hits/$N"; echo ""
}

reproduce_n "$CTRL_CONFIG"    "CONTROL (hpa_gpa=true) "; CTRL_HITS=$HITS
reproduce_n "$NOALIAS_CONFIG" "NOALIAS (hpa_gpa=false)"; NA_HITS=$HITS

echo "=================================================================="
echo " RESULT:  CONTROL $CTRL_HITS/$N    NOALIAS $NA_HITS/$N"
if [ "$CTRL_HITS" -gt 0 ] && [ "$NA_HITS" -eq 0 ]; then
    echo " => the leak REQUIRES hpa_gpa aliasing. Cross-actor reach depends on the manufactured GPA=HPA"
    echo "    collision (the Foreshadow-VM condition) — realism now hinges on whether a guest can reach it."
elif [ "$CTRL_HITS" -gt 0 ] && [ "$NA_HITS" -gt 0 ]; then
    echo " => the leak SURVIVES with aliasing OFF -> it does NOT need the manufactured collision."
    echo "    Much more significant; trace what channel remains (same-VA layout? other aliasing?)."
else
    echo " => CONTROL did not reproduce ($CTRL_HITS/$N) — signal down this session (SMT on after reboot?"
    echo "    run perf_env.sh). Confirm with verify_125257.sh-style reproduce, then re-run."
fi
echo "=================================================================="