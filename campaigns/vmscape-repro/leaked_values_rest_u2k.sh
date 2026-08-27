#!/usr/bin/env bash
#
# leaked_values_rest_u2k.sh — isolate the LEAKED VALUES of each minimized refuzz-u2k violation,
# ONE AT A TIME, in order.
#
# For each violation with a program.min.asm, runs ONLY the input passes (instruction pass OFF):
#   input-diff-pass : minimize the byte-difference between the two counterexample inputs while the
#                     violation holds -> surviving bytes = the leaked values ("Leaked N bytes / Addresses").
#   input-seq-pass  : trim the input set to the minimal reproducing subset.
# Inputs LOADED (RVZR_MINIMIZE_INPUT_DIR). SEQUENTIAL: each run is foreground; the next starts only
# after the current exits (never two against the executor at once).
#
# After each, it decodes the leaked addresses to actor/region (addr = actor*0x4000 + region_off + i*8;
# actor 0=main/host, 1=vmvictim/victim, 2=vm/observer; region main<0x1000, faulty<0x2000, else gpr/simd).
#
# Usage:
#   bash leaked_values_rest_u2k.sh                 # the three new violations (090802 skipped — already done)
#   ONLY="violation-260817-090802" bash leaked_values_rest_u2k.sh   # to (re)do a specific one
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
OUTDIR=$CAMP/out-refuzz-u2k
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
ATTEMPTS=${ATTEMPTS:-1}

[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

if [ -n "${ONLY:-}" ]; then
    WORK=(); for n in $ONLY; do WORK+=("$OUTDIR/$n"); done
else
    WORK=()
    for V in $(ls -d "$OUTDIR"/violation-* 2>/dev/null | sort); do
        [ "$(basename "$V")" = "violation-260817-090802" ] && continue   # already characterized (1 byte @ 0x4640)
        [ -f "$V/program.min.asm" ] && WORK+=("$V")
    done
fi
[ "${#WORK[@]}" -gt 0 ] || { echo "no minimized programs in $OUTDIR"; exit 2; }

decode_addrs () {  # $1 = python-list string like ['0x4640', '0x...']
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
echo " SEQUENTIAL LEAKED-VALUES — ${#WORK[@]} violation(s), input passes, one at a time"
echo "   module : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
echo "=================================================================="
cd "$REPO"

declare -a SUMMARY=()
i=0
for V in "${WORK[@]}"; do
    i=$((i+1)); name=$(basename "$V")
    cfg=$CONFIG; [ -f "$cfg" ] || cfg=$V/reproduce.yaml
    prog=$V/program.min.asm
    inputs=$(ls "$V"/input_*.bin 2>/dev/null | wc -l)
    log=$V/leaked-values.log
    echo ""
    echo ">>>>>>>>>>>>>>>> [$i/${#WORK[@]}] $name  ($(date +%H:%M:%S)) >>>>>>>>>>>>>>>>"
    echo "   program $(basename "$prog"), $inputs inputs (loaded)"

    export RVZR_MINIMIZE_INPUT_DIR="$V"
    # foreground: shell blocks until this run exits before the next violation
    "$PY" revizor.py minimize -s "$BASE" -c "$cfg" -I "$V" \
        -t "$prog" -i "$inputs" -o "$V/program.inputmin.asm" --input-outdir "$V/min-inputs" \
        --num-attempts "$ATTEMPTS" \
        --enable-instruction-pass false \
        --enable-input-diff-pass true \
        --enable-input-seq-pass true 2>&1 | tee "$log"

    leaked=$(grep -oE "Leaked [0-9]+ bytes" "$log" | tail -1)
    addrs=$(grep -oE "Addresses: \[[^]]*\]" "$log" | tail -1 | sed 's/Addresses: //')
    echo "   ---- $name : ${leaked:-'(no Leaked line — did not converge)'} ----"
    if [ -n "$addrs" ]; then
        echo "   leaked addresses $addrs:"
        decode_addrs "$addrs"
    fi
    SUMMARY+=("$name : ${leaked:-no-result}  ${addrs:-}")
done

echo ""
echo "=================================================================="
echo " SUMMARY — leaked values per violation:"
printf '   %s\n' "${SUMMARY[@]}"
echo " (per-run detail saved in each <violation>/leaked-values.log)"
echo "=================================================================="