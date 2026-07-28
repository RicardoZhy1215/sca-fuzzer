#!/usr/bin/env bash
# Side-by-side: a REAL Meltdown gadget vs. this violation's gadget, under the SAME
# contract (ct + delayed-exception-handling) and the SAME fault (present:false).
# Only the gadget structure differs.
#
#   A  real-Meltdown  : fault_load.asm + revizor's own l1tf-p.yaml   (tool authors' Meltdown test)
#   B  real-Meltdown  : fault_load.asm + THIS violation's reproduce.yaml (identical config to C)
#   C  this violation : min.asm        + reproduce.yaml
#
# fault_load.asm leaks the FAULTING load's own value:
#     mov rax,[r14+4096]   ; faulting load (non-present page)
#     ... mov rax,[r14+rax] ; encodes the faulting load's value  -> classic Meltdown/L1TF
# min.asm discards it (sub rbx,rbx) and leaks a separate PRESENT load's value.
#
# EXPECTED on a Meltdown-patched CPU (Intel model >= 140, e.g. w5-3423):
#     A = NO , B = NO   (RDCL_NO: faulting load forwards nothing -> no leak)
#     C = YES           (does not rely on the patched forwarding path)
#
# No sudo. Disable SMT first for clean measurements.
#   bash compare_meltdown.sh          # REPEAT=3 each
#   REPEAT=5 bash compare_meltdown.sh
set -u
REPO=/home/hz25d/beyond_v1_v4/sca-fuzzer
PY=/home/hz25d/miniconda3/envs/specrl/bin/python
HT=rvzr/SpecRL/hierarchical_testing
NAME="${1:-violation-260721-212748}"
REPEAT="${REPEAT:-3}"
V="$HT/$NAME"

MELTDOWN_ASM="tests/x86_tests/asm/fault_load.asm"
MELTDOWN_CFG="tests/x86_tests/configs/l1tf-p.yaml"
MELTDOWN_INC="tests/x86_tests/configs"

cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }
for f in base.json "$MELTDOWN_ASM" "$MELTDOWN_CFG" "$V/reproduce.yaml" "$V/min.asm"; do
    [ -f "$f" ] || { echo "missing: $REPO/$f"; exit 1; }
done
shopt -s nullglob
inputs=( "$V/min-inputs"/min_input*.bin )
shopt -u nullglob
[ ${#inputs[@]} -gt 0 ] || { echo "no min-inputs/min_input*.bin"; exit 1; }
[ "$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)" = "on" ] && \
    echo "WARNING: SMT is ON -> measurements may be noisy. disable it for clean results."

# classify one reproduce run over REPEAT reps; args: label, then full revizor args
run(){
    local label="$1"; shift
    local y=0 n=0 e=0 i log rc
    for i in $(seq 1 "$REPEAT"); do
        log="$V/cmp_${label}_run${i}.log"
        "$PY" ./revizor.py reproduce "$@" >"$log" 2>&1; rc=$?
        if   grep -aqiE 'Input/output error|CalledProcessError|Traceback' "$log"; then e=$((e+1))
        elif grep -aqiE 'Violations detected|Violations: [1-9]' "$log"; then y=$((y+1))
        else n=$((n+1)); fi
    done
    printf '%-26s -> YES=%d NO=%d ERR=%d\n' "$label" "$y" "$n" "$e"
    eval "Y_$label=$y"; eval "E_$label=$e"
}

echo "contract = ct + delayed-exception-handling   fault = present:false   (identical across A/B/C)"
echo "CPU      = $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')   (model $(grep -m1 -E '^model[[:space:]]*:' /proc/cpuinfo | cut -d: -f2 | tr -d ' '))"
echo "repeat   = $REPEAT"
echo "------------------------------------------------------------"
run A_meltdown_l1tfcfg -s base.json -c "$MELTDOWN_CFG" -I "$MELTDOWN_INC" -t "$MELTDOWN_ASM" -n 100
run B_meltdown_reprcfg -s base.json -c "$V/reproduce.yaml"                 -t "$MELTDOWN_ASM" -n 100
run C_this_violation   -s base.json -c "$V/reproduce.yaml"                 -t "$V/min.asm" -i "${inputs[@]}"
echo "------------------------------------------------------------"

ok=1
[ "${Y_A_meltdown_l1tfcfg:-0}" -eq 0 ] && [ "${E_A_meltdown_l1tfcfg:-0}" -eq 0 ] || ok=0
[ "${Y_B_meltdown_reprcfg:-0}" -eq 0 ] && [ "${E_B_meltdown_reprcfg:-0}" -eq 0 ] || ok=0
[ "${Y_C_this_violation:-0}"   -gt 0 ] || ok=0
echo "VERDICT:"
echo "  real Meltdown gadget (A,B): $([ "${Y_A_meltdown_l1tfcfg:-0}" -eq 0 ] && [ "${Y_B_meltdown_reprcfg:-0}" -eq 0 ] && echo 'NO leak  -> classic Meltdown is patched on this CPU (RDCL_NO)' || echo 'leaked -- unexpected; check SMT/logs')"
echo "  this violation (C)        : $([ "${Y_C_this_violation:-0}" -gt 0 ] && echo 'YES  -> reproduces WITHOUT the patched forwarding path' || echo 'NO -- check SMT/logs')"
[ "$ok" = 1 ] && echo "  => clean contrast: same contract+fault, real Meltdown dies, this gadget lives." \
              || echo "  => inconclusive (see ERR / disable SMT and rerun)."
