#!/usr/bin/env bash
# Disambiguate WHY the faulting xchg is required: is it ATOMICITY (LOCK), or just the
# READ/RMW component (a faulting load opens the transient window)?
#
# noxchg (xchg -> plain store) already = NO. Now vary the first (faulting) instruction:
#   load     : mov  rbx, [r14+rdi]        non-atomic, READ-only faulting load
#   lockadd  : lock add  [r14+rdi], rbx   atomic RMW (has read component)
#   lockxadd : lock xadd [r14+rdi], rbx   atomic RMW (has read component)
#   lockinc  : lock inc  qword ptr [r14+rdi]   atomic RMW, no source reg
#
# rbx is zeroed right after (sub rbx,rbx), so clobbering it in these variants is harmless.
#
# Interpretation:
#   load     YES -> it's the faulting LOAD / read component, NOT atomicity  (closer to Meltdown-type)
#   load     NO  + any lock* YES -> ATOMICITY (LOCK) is the essential ingredient (the novel angle)
#   all lock* YES, load NO       -> strong: atomic RMW fault opens the window; plain load does not
#
# No sudo. Disable SMT first. Usage: bash control_atomicity.sh   (REPEAT=3 default)
set -u
REPO=/home/hz25d/beyond_v1_v4/sca-fuzzer
PY=/home/hz25d/miniconda3/envs/specrl/bin/python
HT=rvzr/SpecRL/hierarchical_testing
NAME="${1:-violation-260721-212748}"
REPEAT="${REPEAT:-3}"
V="$HT/$NAME"

cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }
for f in "$V/reproduce.yaml" "$V/min.asm" base.json; do
    [ -f "$f" ] || { echo "missing: $REPO/$f"; exit 1; }
done
shopt -s nullglob
inputs=( "$V/min-inputs"/min_input*.bin )
shopt -u nullglob
[ ${#inputs[@]} -gt 0 ] || { echo "no min-inputs/min_input*.bin"; exit 1; }
[ "$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)" = "on" ] && \
    echo "WARNING: SMT is ON -> measurements may be noisy. disable it for clean controls."

XLINE='xchg qword ptr [r14 + rdi], rbx'
mk() { sed "s|^${XLINE}\$|$1|" "$V/min.asm" > "$V/min_$2.asm"; }
mk 'mov rbx, qword ptr [r14 + rdi]'        load
mk 'lock add qword ptr [r14 + rdi], rbx'   lockadd
mk 'lock xadd qword ptr [r14 + rdi], rbx'  lockxadd
mk 'lock inc qword ptr [r14 + rdi]'        lockinc

run_variant() {  # $1=label $2=asm
    local label="$1" asm="$2" y=0 n=0 e=0 i log rc
    for i in $(seq 1 "$REPEAT"); do
        log="$V/atom_${label}_run${i}.log"
        "$PY" ./revizor.py reproduce -s base.json -c "$V/reproduce.yaml" -t "$V/$asm" -i "${inputs[@]}" >"$log" 2>&1
        rc=$?
        if   grep -aqiE 'Input/output error|CalledProcessError|Traceback' "$log"; then e=$((e+1))
        elif grep -aqiE 'Violations detected|Violations: [1-9]' "$log"; then y=$((y+1))
        else n=$((n+1)); fi
    done
    printf '%-9s (%-34s) -> YES=%d NO=%d ERR=%d\n' "$label" "$asm" "$y" "$n" "$e"
    eval "Y_$label=$y"
}

echo "folder=$NAME  inputs=${#inputs[@]}  repeat=$REPEAT"
echo "first faulting instruction ->"
echo "------------------------------------------------------------"
run_variant baseline min.asm          # xchg, expect YES
run_variant load     min_load.asm     # plain faulting load
run_variant lockadd  min_lockadd.asm
run_variant lockxadd min_lockxadd.asm
run_variant lockinc  min_lockinc.asm
echo "------------------------------------------------------------"
echo "VERDICT:"
if [ "${Y_load:-0}" -gt 0 ]; then
    echo "  load=YES  -> the READ/LOAD component (faulting load) opens the window; atomicity NOT required."
    echo "             => closer to a known Meltdown/fault-shadow-type mechanism."
else
    if [ "${Y_lockadd:-0}" -gt 0 ] || [ "${Y_lockxadd:-0}" -gt 0 ] || [ "${Y_lockinc:-0}" -gt 0 ]; then
        echo "  load=NO but some lock*=YES -> ATOMICITY (LOCK) is the essential ingredient."
        echo "             => faulting atomic RMW opens a transient window a plain faulting load does NOT."
        echo "             => this is the distinctive / likely-underexplored mechanism."
    else
        echo "  load=NO and all lock*=NO -> only xchg specifically works; investigate xchg's uarch path."
    fi
fi
