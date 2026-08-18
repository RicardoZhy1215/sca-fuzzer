#!/usr/bin/env bash
#
# lfence_test_090802.sh — gold-standard speculation test for the 090802 leak.
#
# The observer's retrieval gadget runs AFTER the faulting loads (the transient source) with only an
# mfence between them (L87) — and the leak survives that mfence (mfence orders memory but does NOT
# block speculation). This test ADDS an `lfence` at the same point (lfence DOES block speculation).
#
#   CONTROL = program.min.asm            (mfence only)          -> expect ~10/10
#   LFENCE  = program.min.lfence.asm     (mfence + lfence)      -> the test
#
# Interpretation:
#   control 10/10 + lfence 0/10  -> DEFINITIVELY SPECULATIVE: blocking the observer's speculation
#                                   kills the channel. A genuine transient-execution cross-actor leak.
#   control 10/10 + lfence 10/10 -> survives full serialization at this point -> NOT (or not only)
#                                   the faulty-load->retrieval transient; escalate (fence earlier).
#
# Usage:
#   bash lfence_test_090802.sh              # N=10 each
#   N=20 bash lfence_test_090802.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/reproduce.yaml
CTRL=$V/program.min.asm
LF=$V/program.min.lfence.asm
N=${N:-10}

[ -f "$CTRL" ] || { echo "no minimized program at $CTRL — run minimize_090802.sh first"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# build the lfence variant: add an lfence right after the observer's post-faulty-load mfence (L87),
# i.e. between the transient source (faulting loads) and the retrieval/encode gadget.
awk '
  { print }
  /^mfence[[:space:]]+# noremove/ { print "lfence  # SPECULATION BARRIER (lfence-test)" }
' "$CTRL" > "$LF"

added=$(grep -c "SPECULATION BARRIER" "$LF")
[ "$added" -eq 1 ] || { echo "transform failed: inserted $added lfence lines (expected 1). Is the mfence line present?"; exit 2; }

echo "=================================================================="
echo " LFENCE TEST — 090802: add a speculation barrier after the faulting loads"
echo "   control : $CTRL"
echo "   variant : $LF"
echo "   config  : $CONFIG      inputs: $(ls "$V"/input_*.bin | wc -l) saved      module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "------------------------------------------------------------------"
echo " inserted barrier (context):"
grep -nB6 "SPECULATION BARRIER" "$LF" | sed 's/^/   /'
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

reproduce_n "$CTRL" "CONTROL (mfence only)  "; CTRL_HITS=$HITS
reproduce_n "$LF"   "LFENCE  (spec. blocked)"; LF_HITS=$HITS

echo "=================================================================="
echo " RESULT:  CONTROL $CTRL_HITS/$N    LFENCE $LF_HITS/$N"
if [ "$CTRL_HITS" -gt 0 ] && [ "$LF_HITS" -eq 0 ]; then
    echo " => DEFINITIVELY SPECULATIVE: an lfence between the faulting loads and the retrieval gadget"
    echo "    kills the channel. Genuine transient-execution cross-actor leak (mfence didn't stop it)."
elif [ "$CTRL_HITS" -gt 0 ] && [ "$LF_HITS" -gt 0 ]; then
    echo " => survives serialization at this point. The transient encode is NOT (only) faulty-load->"
    echo "    retrieval. Next: fence right after measurement_start (serialize the whole observer)."
else
    echo " => CONTROL did not reproduce ($CTRL_HITS/$N) — signal down this session. Verify first:"
    echo "    N=10 MIN=$CTRL V=$V CONFIG=$CONFIG bash $CAMP/verify_min_090802.sh   then re-run."
fi
echo "=================================================================="
