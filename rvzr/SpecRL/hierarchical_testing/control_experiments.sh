#!/usr/bin/env bash
# Mechanism controls for the minimized violation (min.asm + min-inputs).
# Determines WHY it leaks: is the fault necessary? is it speculative? is atomicity essential?
#
# Builds 3 variants and reproduces each (REPEAT times), then prints a verdict table.
#   baseline : min.asm            + reproduce.yaml            (should stay YES = sanity)
#   present  : min.asm            + reproduce_present.yaml    (present:false -> true; removes the fault)
#   fenced   : min_fenced.asm     + reproduce.yaml            (LFENCE right after the faulting xchg)
#   noxchg   : min_noxchg.asm     + reproduce.yaml            (xchg -> plain mov; drops atomicity/LOCK)
#
# Interpretation:
#   present  NO  -> the #PF fault is REQUIRED  => genuine fault-shadow transient leak (DEH)
#   fenced   NO  -> LFENCE kills it            => genuinely speculative/transient (not architectural)
#   noxchg   NO  -> atomicity is ESSENTIAL     => the interesting/likely-novel angle (atomic-gated)
#   (any that stay YES flip the corresponding conclusion)
#
# No sudo. Disable SMT first for clean measurements. Usage:
#   bash control_experiments.sh                 # REPEAT=3 each
#   REPEAT=5 bash control_experiments.sh
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
    echo "WARNING: SMT is ON -> measurements may be noisy/EIO. 'echo off | sudo tee /sys/devices/system/cpu/smt/control'"

# --- build variants -------------------------------------------------------------
# present-toggle config: present:false -> present:true (page becomes present, no #PF)
sed 's/present: false/present: true/' "$V/reproduce.yaml" > "$V/reproduce_present.yaml"
# fenced program: LFENCE immediately after the faulting xchg (blocks transient continuation)
awk '{print} /^xchg qword ptr \[r14 \+ rdi\], rbx/{print "lfence"}' "$V/min.asm" > "$V/min_fenced.asm"
# non-atomic program: xchg -> mov (same store effect: xchg read result is discarded by sub rbx,rbx)
sed 's/^xchg qword ptr \[r14 + rdi\], rbx/mov qword ptr [r14 + rdi], rbx/' "$V/min.asm" > "$V/min_noxchg.asm"

# --- reproduce helper: prints YES/NO/ERR over REPEAT runs -----------------------
run_variant() {  # $1=label  $2=asm  $3=cfg
    local label="$1" asm="$2" cfg="$3" y=0 n=0 e=0 i log rc r
    for i in $(seq 1 "$REPEAT"); do
        log="$V/ctrl_${label}_run${i}.log"
        "$PY" ./revizor.py reproduce -s base.json -c "$V/$cfg" -t "$V/$asm" -i "${inputs[@]}" >"$log" 2>&1
        rc=$?
        if   grep -aqiE 'Input/output error|CalledProcessError|Traceback' "$log"; then e=$((e+1))
        elif grep -aqiE 'Violations detected|Violations: [1-9]' "$log"; then y=$((y+1))
        else n=$((n+1)); fi
    done
    printf '%-9s asm=%-16s cfg=%-22s -> YES=%d NO=%d ERR=%d\n' "$label" "$asm" "$cfg" "$y" "$n" "$e"
    eval "V_$label=$y"; eval "N_$label=$n"; eval "E_$label=$e"
}

echo "folder=$NAME  inputs=${#inputs[@]}  repeat=$REPEAT"
echo "------------------------------------------------------------"
run_variant baseline min.asm         reproduce.yaml
run_variant present  min.asm         reproduce_present.yaml
run_variant fenced   min_fenced.asm  reproduce.yaml
run_variant noxchg   min_noxchg.asm  reproduce.yaml
echo "------------------------------------------------------------"

verdict() {  # $1=label $2=text-if-NO $3=text-if-YES
    local y n; eval "y=\$V_$1"; eval "n=\$N_$1"
    if [ "$y" -eq 0 ] && [ "$n" -gt 0 ]; then echo "  [$1] NO  -> $2"
    elif [ "$y" -gt 0 ];                  then echo "  [$1] YES -> $3"
    else echo "  [$1] inconclusive (errors) -- disable SMT and rerun"; fi
}
echo "VERDICT:"
verdict present "fault #PF is REQUIRED  => genuine fault-shadow transient leak (DEH)" \
                "leak persists WITHOUT the fault => architectural or not fault-gated; rethink DEH story"
verdict fenced  "LFENCE kills it => genuinely SPECULATIVE/transient" \
                "survives LFENCE => NOT (purely) speculative; likely architectural or a contract-model gap"
verdict noxchg  "atomicity is ESSENTIAL => atomic-gated fault-shadow store leak (the novel angle)" \
                "works with plain mov too => atomicity NOT essential; any faulting store suffices"
