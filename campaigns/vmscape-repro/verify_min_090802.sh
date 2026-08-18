#!/usr/bin/env bash
#
# verify_min_090802.sh — confirm the MINIMIZED program (program.min.asm, 70 instrs) still triggers the
# refuzz-u2k violation-260817-090802, i.e. minimization preserved the leak.
#
# Reproduces program.min.asm N times with the campaign config (config-refuzz-u2k.yaml, threshold 0.5)
# and the 100 saved inputs (min-inputs is empty — instruction-pass-only run). exit 1 = violation, 0 =
# clean. Compare to the original (ORIG=1). The original reproduced 10/10, so the minimized should too.
#
# Usage:
#   bash verify_min_090802.sh            # N=10 on program.min.asm
#   N=20 bash verify_min_090802.sh
#   ORIG=1 bash verify_min_090802.sh     # also reproduce the ORIGINAL 236-instr program, for comparison
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/reproduce.yaml
MIN=${MIN:-$V/program.min.asm}
N=${N:-10}
ORIG=${ORIG:-0}

count_instr () { grep -vE '^\s*(#|\.|$)' "$1" 2>/dev/null | grep -cvE '^\s*$'; }

[ -f "$MIN" ] || { echo "no minimized program at $MIN — run minimize_090802.sh first"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# prefer the minimized inputs if the input passes ran; else the 100 saved inputs the program reproduced with
if ls "$V/min-inputs"/*.bin >/dev/null 2>&1; then
    IN_GLOB=("$V/min-inputs"/*.bin); IN_SRC="min-inputs ($(ls "$V/min-inputs"/*.bin | wc -l))"
else
    IN_GLOB=("$V"/input_*.bin);      IN_SRC="saved inputs ($(ls "$V"/input_*.bin | wc -l)) [no input passes ran]"
fi

echo "=================================================================="
echo " VERIFY minimized reproduces — violation-260817-090802"
echo "   minimized : $MIN  ($(count_instr "$MIN") instrs)"
echo "   original  : $V/program.asm  ($(count_instr "$V/program.asm") instrs)"
echo "   config    : $CONFIG   ($(grep -E 'analyser_stat_threshold' "$CONFIG" | awk '{print $2}') thr)"
echo "   inputs    : $IN_SRC"
echo "   module    : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)   (want clean 1CD139EF)"
echo "=================================================================="
cd "$REPO"

reproduce_n () {   # $1 = program, $2 = label ; sets HITS
    local prog=$1 label=$2 hits=0 rc b a
    echo "--- $label : $(basename "$prog")  (x$N) ---"
    for i in $(seq 1 "$N"); do
        b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$prog" -i "${IN_GLOB[@]}" >/dev/null 2>&1
        rc=$?
        a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        if [ "$a" -gt "$b" ]; then
            echo "  !! hpa_gpa EPT COLLISION — reload: bash $CAMP/reload_until_ok.sh"; exit 3
        fi
        [ "$rc" -eq 1 ] && hits=$((hits+1))
        printf "  run %2d: exit=%d  %s\n" "$i" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    done
    HITS=$hits
    echo "  => $label reproduced $hits/$N"; echo ""
}

reproduce_n "$MIN" "MINIMIZED"; MIN_HITS=$HITS
if [ "$ORIG" = "1" ]; then reproduce_n "$V/program.asm" "ORIGINAL"; ORIG_HITS=$HITS; fi

echo "=================================================================="
echo " RESULT: MINIMIZED reproduced $MIN_HITS/$N${ORIG_HITS:+   (ORIGINAL $ORIG_HITS/$N)}"
if [ "$MIN_HITS" -gt 0 ]; then
    echo " => the 70-instr minimized program STILL triggers the violation ✓ (minimization preserved the leak)"
else
    echo " => minimized did NOT reproduce. Confirm the original still does (bash replay_refuzz_u2k.sh),"
    echo "    module is clean 1CD139EF, and perf_env.sh is applied; then raise N."
fi
echo "=================================================================="
