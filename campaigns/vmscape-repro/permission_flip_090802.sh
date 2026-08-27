#!/usr/bin/env bash
#
# permission_flip_090802.sh — root-cause doc guideline 4 (Change Memory Permissions).
#
# The observer's data page is read-only (data_properties: writable: false), so its store at L101
# (mov byte ptr [r14+rdi], al -> faulty area 0xab82) FAULTS. This test flips that page to writable,
# which removes the write-fault, and checks whether the violation disappears.
#
#   CONTROL = config-refuzz-u2k.yaml       (writable: false -> store faults)   -> expect ~10/10
#   FLIPPED = writable: true               (store no longer faults)            -> the test
#
# Interpretation:
#   control 10/10 + flipped 0/10  -> the WRITE-FAULT (permission) is the trigger. Fault-driven leak
#                                    (a store to a read-only page opening the speculative window).
#   control 10/10 + flipped 10/10 -> the fault is NOT the trigger. The speculation comes from
#                                    something else (SSB / scheduler / data-dependent), keep looking.
#
# Same program + same inputs both times; only the page permission changes.
#
# Usage:
#   bash permission_flip_090802.sh              # N=10 each, on program.min.asm
#   N=20 bash permission_flip_090802.sh
#   PROG=$V/program.asm bash permission_flip_090802.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CTRL_CONFIG=${CTRL_CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CTRL_CONFIG" ] || CTRL_CONFIG=$V/reproduce.yaml
FLIP_CONFIG=$CAMP/config-090802-writable.yaml
PROG=${PROG:-$V/program.min.asm}
N=${N:-10}

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# derive the flipped config: writable: false -> true (change nothing else)
sed 's/writable: false/writable: true/' "$CTRL_CONFIG" > "$FLIP_CONFIG"
if ! grep -q "writable: true" "$FLIP_CONFIG"; then
    echo "sed failed — 'writable: false' not found in $CTRL_CONFIG; check the actor data_properties"; exit 2
fi

echo "=================================================================="
echo " PERMISSION FLIP — 090802: observer page read-only -> writable (remove the store-fault)"
echo "   program : $PROG  ($(grep -vE '^\s*(#|\.|$)' "$PROG" | grep -cvE '^\s*$') instrs)"
echo "   inputs  : $(ls "$V"/input_*.bin | wc -l) saved   module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "   CONTROL : writable:false ($CTRL_CONFIG)"
echo "   FLIPPED : writable:true  ($FLIP_CONFIG)"
echo "=================================================================="
cd "$REPO"

reproduce_n () {   # $1 program, $2 config, $3 label ; sets HITS
    local prog=$1 cfg=$2 label=$3 hits=0 rc b a
    echo "--- $label : reproduce x$N ---"
    for i in $(seq 1 "$N"); do
        b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        "$PY" revizor.py reproduce -s "$BASE" -c "$cfg" -I "$V" -t "$prog" -i "$V"/input_*.bin >/dev/null 2>&1
        rc=$?
        a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        if [ "$a" -gt "$b" ]; then echo "  !! hpa_gpa EPT COLLISION — reload: bash $CAMP/reload_until_ok.sh"; exit 3; fi
        [ "$rc" -eq 1 ] && hits=$((hits+1))
        printf "  run %2d: exit=%d  %s\n" "$i" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    done
    HITS=$hits; echo "  => $label reproduced $hits/$N"; echo ""
}

reproduce_n "$PROG" "$CTRL_CONFIG" "CONTROL (writable:false)"; CTRL_HITS=$HITS
reproduce_n "$PROG" "$FLIP_CONFIG" "FLIPPED (writable:true) "; FLIP_HITS=$HITS

echo "=================================================================="
echo " RESULT:  CONTROL $CTRL_HITS/$N    FLIPPED $FLIP_HITS/$N"
if [ "$CTRL_HITS" -gt 0 ] && [ "$FLIP_HITS" -eq 0 ]; then
    echo " => VIOLATION DISAPPEARS when the page is writable. The WRITE-FAULT (permission) is the"
    echo "    trigger -> fault-driven speculative leak (store to a read-only page opens the window)."
elif [ "$CTRL_HITS" -gt 0 ] && [ "$FLIP_HITS" -gt 0 ]; then
    echo " => leak SURVIVES with a writable page -> the fault is NOT the trigger. Speculation source"
    echo "    is elsewhere (SSB / scheduler / data dependency). Next: fence pass to localize it."
else
    echo " => CONTROL did not reproduce ($CTRL_HITS/$N) — signal down this session. Verify first:"
    echo "    N=10 MIN=$PROG V=$V bash $CAMP/verify_min_090802.sh   then re-run."
fi
echo "=================================================================="
