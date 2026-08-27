#!/usr/bin/env bash
#
# leaked_values_noalias_090802.sh — isolate the leaked byte with hpa_gpa OFF (separate physical).
#
# Runs the input-diff + input-seq passes on program.min.asm using a config with
# x86_enable_hpa_gpa_collisions:false, so the guests' data pages are DIFFERENT host-physical pages.
# If it still leaks the SAME byte (vmvictim MAIN 0x640) as the aliased run, that's a cross-actor
# VALUE forward under genuine physical separation — no shared memory involved.
#
#   same byte 0x4640 (actor1/vmvictim/main/0x640) -> confirmed cross-actor value forward (separate HPA)
#   different byte / no result                     -> the off-case leak is a different phenomenon
#
# NB: the hpa_gpa-off signal is weaker (~4/5); run perf_env.sh first, and re-run if it doesn't converge.
#
# Usage:
#   bash leaked_values_noalias_090802.sh
#   ATTEMPTS=2 bash leaked_values_noalias_090802.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
SRC_CONFIG=${SRC_CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$SRC_CONFIG" ] || SRC_CONFIG=$V/reproduce.yaml
NOALIAS_CONFIG=$CAMP/config-090802-noalias.yaml
PROG=${PROG:-$V/program.min.asm}
INDIR=${INDIR:-$V/min-inputs-noalias}
OUT=${OUT:-$V/program.inputmin.noalias.asm}
INPUTS=${INPUTS:-$(ls "$V"/input_*.bin 2>/dev/null | wc -l)}
ATTEMPTS=${ATTEMPTS:-1}
LOG=$V/leaked-values-noalias.log

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# derive the no-aliasing config (only hpa_gpa flips)
sed 's/^x86_enable_hpa_gpa_collisions:.*/x86_enable_hpa_gpa_collisions: false/' "$SRC_CONFIG" > "$NOALIAS_CONFIG"
grep -q "x86_enable_hpa_gpa_collisions: false" "$NOALIAS_CONFIG" || { echo "sed failed on hpa_gpa key"; exit 2; }

decode_addrs () {   # $1 = "['0x4640', ...]"
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
echo " LEAKED VALUES with hpa_gpa OFF (separate physical) — 090802"
echo "   program : $PROG   config: $NOALIAS_CONFIG (hpa_gpa=false)"
echo "   inputs  : $INPUTS loaded   module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  ucode: $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)"
echo "   SMT: $(cat /sys/devices/system/cpu/smt/control 2>/dev/null) gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)  (perf_env.sh for a clean run)"
echo "=================================================================="
mkdir -p "$INDIR"
export RVZR_MINIMIZE_INPUT_DIR="$V"
cd "$REPO"
"$PY" revizor.py minimize -s "$BASE" -c "$NOALIAS_CONFIG" -I "$V" \
    -t "$PROG" -i "$INPUTS" -o "$OUT" --input-outdir "$INDIR" --num-attempts "$ATTEMPTS" \
    --enable-instruction-pass false \
    --enable-input-diff-pass true \
    --enable-input-seq-pass true 2>&1 | tee "$LOG"

echo ""
echo "------------------------------------------------------------------"
leaked=$(grep -oE "Leaked [0-9]+ bytes" "$LOG" | tail -1)
addrs=$(grep -oE "Addresses: \[[^]]*\]" "$LOG" | tail -1 | sed 's/Addresses: //')
echo " RESULT (hpa_gpa OFF): ${leaked:-'(no Leaked line — did not converge; run perf_env + retry)'}"
if [ -n "$addrs" ]; then
    echo " leaked addresses $addrs:"
    decode_addrs "$addrs"
    if echo "$addrs" | grep -q "0x4640"; then
        echo " => SAME byte (0x4640 = vmvictim/main/0x640) as the aliased run."
        echo "    Cross-actor VALUE forward confirmed with SEPARATE physical memory."
    else
        echo " => DIFFERENT byte than the aliased 0x4640 — the off-case leak may be a different path."
    fi
fi
echo "=================================================================="