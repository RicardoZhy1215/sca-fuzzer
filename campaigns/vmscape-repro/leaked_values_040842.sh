#!/usr/bin/env bash
#
# leaked_values_040842.sh — isolate the LEAKED VALUES of violation-260825-040842 (both guests KERNEL).
#
# Runs ONLY the input passes (instruction pass OFF): input-diff (minimize the byte-diff between the
# #39/#89 counterexample while the violation holds -> surviving bytes = leaked values) + input-seq.
# Inputs LOADED (RVZR_MINIMIZE_INPUT_DIR). Runs on program.min.asm if present (minimize_040842.sh),
# else program.asm. Then decodes each leaked address to actor/region.
#
#   addr = actor*0x4000 + region_off + i*8 ; actor 0=main/host, 1=vmvictim, 2=vm/observer;
#   region: main<0x1000, faulty<0x2000, else gpr/simd.
#
# Usage:
#   bash leaked_values_040842.sh
#   PROG=$V/program.asm bash leaked_values_040842.sh   # force the original program
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$REPO/violation-260825-040842}
CONFIG=${CONFIG:-$V/reproduce.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/org-config.yaml
if [ -z "${PROG:-}" ]; then
    if [ -f "$V/program.min.asm" ]; then PROG=$V/program.min.asm; else PROG=$V/program.asm; fi
fi
OUT=${OUT:-$V/program.inputmin.asm}
INDIR=${INDIR:-$V/min-inputs}
INPUTS=${INPUTS:-$(ls "$V"/input_*.bin 2>/dev/null | wc -l)}
ATTEMPTS=${ATTEMPTS:-1}
LOG=$V/leaked-values.log

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

decode_addrs () {   # $1 = "['0x...', ...]"
    "$PY" - "$1" <<'EOF'
import sys, ast
try: addrs = ast.literal_eval(sys.argv[1])
except Exception: addrs = []
A={0:"main (HOST)",1:"vmvictim (VICTIM)",2:"vm (observer)"}
for a in addrs:
    v=int(a,16); actor=v//0x4000; within=v%0x4000
    reg = "main   (off 0x%03x)"%within if within<0x1000 else ("faulty (off 0x%03x)"%(within-0x1000) if within<0x2000 else "gpr/simd (off 0x%03x)"%(within-0x2000))
    print("     %s -> actor %d = %-18s %s"%(a, actor, A.get(actor,"?"), reg))
EOF
}

echo "=================================================================="
echo " LEAKED VALUES — violation-260825-040842 (both guests KERNEL)"
echo "   program : $PROG"
echo "   config  : $CONFIG   ($(grep -E 'analyser_stat_threshold' "$CONFIG" | awk '{print $2}') thr)"
echo "   inputs  : $INPUTS loaded   out-dir: $INDIR   counterexample: #39 vs #89"
echo "   module  : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)   ucode: $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)"
echo "=================================================================="
mkdir -p "$INDIR"
export RVZR_MINIMIZE_INPUT_DIR="$V"
cd "$REPO"
"$PY" revizor.py minimize -s "$BASE" -c "$CONFIG" -I "$V" \
    -t "$PROG" -i "$INPUTS" -o "$OUT" --input-outdir "$INDIR" --num-attempts "$ATTEMPTS" \
    --enable-instruction-pass false \
    --enable-input-diff-pass true \
    --enable-input-seq-pass true 2>&1 | tee "$LOG"

echo ""
echo "------------------------------------------------------------------"
leaked=$(grep -oE "Leaked [0-9]+ bytes" "$LOG" | tail -1)
addrs=$(grep -oE "Addresses: \[[^]]*\]" "$LOG" | tail -1 | sed 's/Addresses: //')
echo " RESULT: ${leaked:-'(no Leaked line — did not converge; run perf_env + retry)'}"
if [ -n "$addrs" ]; then
    echo " leaked addresses $addrs:"
    decode_addrs "$addrs"
fi
echo "=================================================================="