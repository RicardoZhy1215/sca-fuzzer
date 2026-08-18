#!/usr/bin/env bash
#
# verify_min.sh — confirm the MINIMIZED program still triggers the violation.
#
# Reproduces program.min.asm (minimize's output) N times, with the SAME config + saved inputs that
# minimize checked against, and reports the rate. If it reproduces, the minimization preserved the
# leak. Because the signal drifts/weak, run N>=10 and compare to the original program's rate.
#
# Usage (defaults = the u2k minimized program):
#   bash verify_min.sh                 # N=10 on program.min.asm
#   N=20 bash verify_min.sh
#   MIN=<path> V=<dir> CONFIG=<yaml> bash verify_min.sh
#   ORIG=1 bash verify_min.sh          # also reproduce the ORIGINAL program N times, for comparison
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-fuzz-u2k/violation-260814-201415}
CONFIG=${CONFIG:-$CAMP/config-fuzz-u2k.yaml}
MIN=${MIN:-$V/program.min.asm}
N=${N:-10}
ORIG=${ORIG:-0}

count_instr () { grep -vE '^\s*(#|\.|$)' "$1" 2>/dev/null | grep -cvE '^\s*$'; }

[ -f "$MIN" ] || { echo "no minimized program at $MIN — run minimize_violation.sh first"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# prefer the MINIMIZED inputs (min-inputs, produced only if input passes ran); else the saved inputs
# minimize actually validated the minimized program against.
if ls "$V/min-inputs"/*.bin >/dev/null 2>&1; then
    IN_GLOB=("$V/min-inputs"/*.bin); IN_SRC="min-inputs ($(ls "$V/min-inputs"/*.bin | wc -l))"
else
    IN_GLOB=("$V"/input_*.bin);      IN_SRC="saved inputs ($(ls "$V"/input_*.bin | wc -l)) [min-inputs empty — no input passes ran]"
fi

echo "=================================================================="
echo " VERIFY minimized program reproduces the violation"
echo "   minimized : $MIN  ($(count_instr "$MIN") instrs)"
echo "   original  : $V/program.asm  ($(count_instr "$V/program.asm") instrs)"
echo "   inputs    : $IN_SRC"
echo "   config    : $CONFIG"
echo "   module    : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "=================================================================="

reproduce_n () {  # $1 = program, $2 = label ; sets HITS
    local prog=$1 label=$2 hits=0 rc bA aA
    echo "--- $label : $prog  (x$N) ---"
    for i in $(seq 1 "$N"); do
        bA=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$prog" -i "${IN_GLOB[@]}" >/dev/null 2>&1
        rc=$?
        aA=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        if [ "$aA" -gt "$bA" ]; then
            echo "  !! hpa_gpa EPT COLLISION — reload: bash $CAMP/reload_until_ok.sh"; exit 3
        fi
        [ "$rc" -eq 1 ] && hits=$((hits+1))
        printf "  run %2d: exit=%d  %s\n" "$i" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || echo clean)"
    done
    HITS=$hits
    echo "  => $label reproduced $hits/$N"
    echo ""
}

reproduce_n "$MIN" "MINIMIZED"; MIN_HITS=$HITS
if [ "$ORIG" = "1" ]; then reproduce_n "$V/program.asm" "ORIGINAL"; ORIG_HITS=$HITS; fi

echo "=================================================================="
echo " RESULT: MINIMIZED reproduced $MIN_HITS/$N${ORIG_HITS:+   (ORIGINAL $ORIG_HITS/$N)}"
if [ "$MIN_HITS" -gt 0 ]; then
    echo " => the minimized program STILL triggers the violation ✓ (minimization preserved the leak)"
else
    echo " => minimized did NOT reproduce this round. Signal is weak/drifting — raise N, and first"
    echo "    confirm the signal is up:  N=5 bash $CAMP/replay_fuzz_u2k.sh"
fi
echo "=================================================================="
