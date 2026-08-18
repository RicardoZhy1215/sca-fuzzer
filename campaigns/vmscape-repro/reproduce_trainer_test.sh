#!/usr/bin/env bash
#
# reproduce_trainer_test.sh — per-test-case CAUSAL test for a saved residue violation:
# is the leak caused by the victim's BTB TRAINING (its indirect jmp) or by the random_instructions?
#
# Uses `revizor.py reproduce` (replays the exact saved program + its 100 inputs; exit 1 = violation,
# 0 = clean) in an A/B:
#   A) ORIGINAL   : the saved program.asm as-is                         -> expected VIOLATION
#   B) NO-TRAINER : same program, victim's trainer 'jmp rax' -> two nops -> expected CLEAN
#
# The swap is BYTE-EXACT: jmp rax = FF E0 = 2 bytes, two nops = 90 90 = 2 bytes, so every other
# address is unchanged. Architecture is identical (the victim falls through into the very gadget its
# jmp targeted) — the ONLY thing removed is the indirect branch that trains the BTB. The observer's
# 'jmp rbx' is left intact. So:
#   A violates AND B clean  -> the training is causally necessary  => BTB RESIDUE (not random insns).
#   A and B both violate    -> leak survives without training      => generic transient (random insns).
#
# Needs the clean module loaded (1CD139EF) and the low-latency env pinned (perf_env.sh).
#
# Usage:
#   bash reproduce_trainer_test.sh                 # default violation dir, 3x each
#   REPEAT=5 bash reproduce_trainer_test.sh
#   V=<other violation dir> bash reproduce_trainer_test.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-residue/violation-260814-035332}
CONFIG=$V/reproduce.yaml
REPEAT=${REPEAT:-3}
ORIG=$V/program.asm
NOP=$CAMP/program-notrain.asm

[ -f "$ORIG" ] || { echo "no program.asm in $V"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
grep -qE "^\s*jmp rax" "$ORIG" || { echo "no trainer 'jmp rax' found in $ORIG"; exit 2; }

echo "=================================================================="
echo " TRAINER CAUSAL TEST (reproduce A/B)   REPEAT=$REPEAT"
echo "   violation : $V"
echo "   config    : $CONFIG"
echo "   module    : $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "=================================================================="

# build the byte-exact trainer-stripped program (only the victim 'jmp rax' -> nop; nop)
sed -E 's/^([[:space:]]*)jmp rax.*/\1nop\n\1nop/' "$ORIG" > "$NOP"
echo "[strip] $NOP  (diff vs original):"
diff "$ORIG" "$NOP" | sed 's/^/   /'
echo ""

reproduce_n () {  # $1=label  $2=program ; sets HITS
    local label=$1 prog=$2 hits=0 rc
    echo "--- $label : $prog  (x$REPEAT) ---"
    for i in $(seq 1 "$REPEAT"); do
        "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$prog" -i "$V"/input_*.bin \
            >/dev/null 2>&1
        rc=$?
        case "$rc" in
            1) hits=$((hits+1)); echo "   run $i: exit=1  VIOLATION" ;;
            0) echo "   run $i: exit=0  clean" ;;
            *) echo "   run $i: exit=$rc ERROR (parse/executor)" ;;
        esac
    done
    HITS=$hits
    echo "   => $label: $hits/$REPEAT violated"
    echo ""
}

reproduce_n "ORIGINAL (trainer present)"   "$ORIG"; ORIG_HITS=$HITS
reproduce_n "NO-TRAINER (jmp rax -> nops)" "$NOP";  NOP_HITS=$HITS

echo "=================================================================="
echo " RESULT (x$REPEAT):   ORIGINAL=$ORIG_HITS/$REPEAT    NO-TRAINER=$NOP_HITS/$REPEAT"
if   [ "$ORIG_HITS" -gt 0 ] && [ "$NOP_HITS" -eq 0 ]; then
    echo " => trainer NECESSARY  ==>  violation is BTB TRAINING (residue), NOT random_instructions."
elif [ "$ORIG_HITS" -gt 0 ] && [ "$NOP_HITS" -gt 0 ]; then
    echo " => leak survives without the trainer  ==>  generic transient (random_instructions), NOT residue."
else
    echo " => ORIGINAL didn't reproduce this round (marginal signal) — raise REPEAT and retry."
fi
echo "=================================================================="
