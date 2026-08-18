#!/usr/bin/env bash
#
# leaked_values_090802.sh — isolate the LEAKED VALUES of refuzz-u2k violation-260817-090802.
#
# Runs ONLY the INPUT passes (instruction pass OFF, program unchanged):
#   input-diff-pass : minimize the byte-difference between the two counterexample inputs (#34,#84)
#                     while the violation still holds -> the bytes that SURVIVE are the leaked values.
#   input-seq-pass  : trim the input set to the minimal reproducing subset.
#
# Feasible now because 090802 reproduces 10/10 at threshold 0.5 (each reduction is validated against
# the violation, which now holds reliably). Inputs are LOADED (RVZR_MINIMIZE_INPUT_DIR), so the
# [#34,#84] counterexample is always present.
#
# Runs on program.min.asm if you already minimized (bash minimize_090802.sh), else program.asm.
#
# Usage:
#   bash leaked_values_090802.sh                       # on program.min.asm if present, else program.asm
#   PROG=$V/program.asm bash leaked_values_090802.sh   # force the original program
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/reproduce.yaml
# default program: the minimized one if it exists (smaller, faster), else the original
if [ -z "${PROG:-}" ]; then
    if [ -f "$V/program.min.asm" ]; then PROG=$V/program.min.asm; else PROG=$V/program.asm; fi
fi
OUT=${OUT:-$V/program.inputmin.asm}       # -o is required; program is unchanged (instruction pass off)
INDIR=${INDIR:-$V/min-inputs}
INPUTS=${INPUTS:-$(ls "$V"/input_*.bin 2>/dev/null | wc -l)}
ATTEMPTS=${ATTEMPTS:-1}

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

echo "=================================================================="
echo " LEAKED VALUES — input-diff + input-seq passes on violation-260817-090802"
echo "   program : $PROG"
echo "   config  : $CONFIG   ($(grep -E 'analyser_stat_threshold' "$CONFIG" | awk '{print $2}') thr)"
echo "   inputs  : $INPUTS loaded   out-dir: $INDIR   counterexample: #34 vs #84"
echo "   module  : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)   (want clean 1CD139EF)"
echo "=================================================================="
mkdir -p "$INDIR"
export RVZR_MINIMIZE_INPUT_DIR="$V"       # load the exact saved inputs (incl. #84)
cd "$REPO"
"$PY" revizor.py minimize -s "$BASE" -c "$CONFIG" -I "$V" \
    -t "$PROG" -i "$INPUTS" -o "$OUT" --input-outdir "$INDIR" --num-attempts "$ATTEMPTS" \
    --enable-instruction-pass false \
    --enable-input-diff-pass true \
    --enable-input-seq-pass true
RC=$?
echo "------------------------------------------------------------------"
echo " minimized inputs written to $INDIR:"
ls -l "$INDIR"/*.bin 2>/dev/null | awk '{print "   "$NF}' | head

n=$(ls "$INDIR"/*.bin 2>/dev/null | wc -l)
if [ "$n" -ge 2 ]; then
    a=$(ls "$INDIR"/*.bin | sed -n 1p); b=$(ls "$INDIR"/*.bin | sed -n 2p)
    sz=$(stat -c%s "$a"); per=$(( sz / 3 ))
    echo ""
    echo " === LEAKED VALUES: bytes that still differ between the two counterexample inputs ==="
    echo "   ($a  vs  $b)   [per-actor region = $per B]"
    cmp -l "$a" "$b" 2>/dev/null | awk -v per="$per" '
      { off=$1-1;
        reg = (off<per?"main(host)":(off<2*per?"vmvictim(GUEST victim)":"vm(observer)"));
        printf "   off %-6d [%s] : %o vs %o\n", off, reg, $2, $3;
        if (off<per) m++; else if (off<2*per) vv++; else vm++; }
      END {
        printf "\n   totals: main=%d  vmvictim=%d  vm=%d  differing bytes\n", m+0, vv+0, vm+0;
        printf "   -> the surviving vmvictim/main bytes ARE the leaked values (vm should stay 0).\n"; }' | head -80
else
    echo " (no minimized inputs produced — PASS 1 didn't reproduce. Confirm 090802 still reproduces:"
    echo "  bash $CAMP/replay_refuzz_u2k.sh   and that perf_env.sh is applied.)"
fi
echo "=================================================================="
exit "$RC"
