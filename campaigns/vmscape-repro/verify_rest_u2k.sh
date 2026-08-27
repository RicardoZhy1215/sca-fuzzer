#!/usr/bin/env bash
#
# verify_rest_u2k.sh — confirm each MINIMIZED refuzz-u2k program still triggers its violation.
#
# STRICTLY SEQUENTIAL: a for-loop; each `reproduce` runs in the foreground (no '&'), so the shell
# blocks until it exits before the next run — never two against the executor at once.
#
# For every out-refuzz-u2k/violation-* that has a program.min.asm, it reproduces program.min.asm
# N times (default 5) with the violation's saved inputs and config-refuzz-u2k.yaml (threshold 0.5),
# and reports hits/N. This covers all four (incl. the already-minimized 090802 as a baseline).
# exit 1 = violation reproduced, 0 = clean.
#
# Usage:
#   bash verify_rest_u2k.sh                 # N=5 each, all minimized violations
#   N=10 bash verify_rest_u2k.sh
#   ONLY="violation-260817-111402 violation-260817-121434 violation-260817-125257" bash verify_rest_u2k.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
OUTDIR=$CAMP/out-refuzz-u2k
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
N=${N:-5}

[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
count_instr () { grep -vE '^\s*(#|\.|$)' "$1" 2>/dev/null | grep -cvE '^\s*$'; }

# work list: violations that have a minimized program (oldest-first for determinism)
if [ -n "${ONLY:-}" ]; then
    WORK=(); for n in $ONLY; do WORK+=("$OUTDIR/$n"); done
else
    WORK=()
    for V in $(ls -d "$OUTDIR"/violation-* 2>/dev/null | sort); do
        [ -f "$V/program.min.asm" ] && WORK+=("$V")
    done
fi
[ "${#WORK[@]}" -gt 0 ] || { echo "no minimized programs found in $OUTDIR (run minimize first)"; exit 2; }

echo "=================================================================="
echo " SEQUENTIAL VERIFY — ${#WORK[@]} minimized violation(s), reproduce x$N each, one at a time"
echo "   module : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
echo "   config : $CONFIG"
echo "=================================================================="
cd "$REPO"

declare -a SUMMARY=()
i=0
for V in "${WORK[@]}"; do
    i=$((i+1)); name=$(basename "$V")
    cfg=$CONFIG; [ -f "$cfg" ] || cfg=$V/reproduce.yaml
    min=$V/program.min.asm
    echo ""
    echo ">>>>>>>>>>>>>>>> [$i/${#WORK[@]}] $name  ($(date +%H:%M:%S)) >>>>>>>>>>>>>>>>"
    echo "   minimized $(count_instr "$min") instrs (orig $(count_instr "$V/program.asm")), $(ls "$V"/input_*.bin|wc -l) inputs"
    hits=0
    for r in $(seq 1 "$N"); do
        # foreground: shell blocks until this reproduce exits before the next run
        "$PY" revizor.py reproduce -s "$BASE" -c "$cfg" -I "$V" -t "$min" -i "$V"/input_*.bin >/dev/null 2>&1
        rc=$?
        [ "$rc" -eq 1 ] && hits=$((hits+1))
        printf "   run %d: exit=%d  %s\n" "$r" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    done
    echo "   => $name reproduced $hits/$N"
    SUMMARY+=("$name : $hits/$N  ($(count_instr "$min") instrs)")
done

echo ""
echo "=================================================================="
echo " SUMMARY — minimized programs reproduce rate (x$N each):"
printf '   %s\n' "${SUMMARY[@]}"
echo "=================================================================="