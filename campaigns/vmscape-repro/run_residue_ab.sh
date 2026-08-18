#!/usr/bin/env bash
#
# run_residue_ab.sh — is residue-585 GENUINE BTB residue, or a generic transient?
#
# A/B at large N, same config (config.yaml, F+R + hpa_gpa), residue template with vs without the
# vmvictim "trainer" (its indirect jmp that trains the BTB):
#   A) template.asm                  — WITH trainer  (the original residue setup)
#   B) template-residue-notrain.asm  — NO trainer    (vmvictim does its preamble, then returns)
#
# Interpretation:
#   A leaks, B == 0  -> the trainer is NECESSARY -> genuine cross-guest BTB residue.
#   A and B both leak at similar rates -> the trainer is irrelevant -> generic transient (not residue).
#
# NOTE: this script does NOT touch the kernel module — load the build you want yourself first
# (the clean, alias-free 1CD139EF is the right one for this test). residue leaks are RARE (the
# original hit at test case 585), so N must be well past that to be fair; default N=2000 (~25 min/run).
#
# Usage:
#   bash run_residue_ab.sh            # N=2000 each (long)
#   N=800 bash run_residue_ab.sh      # quicker, still past the 585 mark
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
CONFIG=$CAMP/config.yaml
N=${N:-2000}; I=${I:-50}

echo "loaded executor: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want the clean 1CD139EF)"
[ -e /sys/rvzr_executor ] || { echo "no /sys/rvzr_executor — load the module first"; exit 2; }

run_one () {  # $1 = label, $2 = template, $3 = outdir
    echo "=================================================================="
    echo " [$1]  $2   (N=$N x I=$I)   $(date +%H:%M:%S)"
    echo "=================================================================="
    mkdir -p "$3"
    cd "$REPO"
    "$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$2" -n "$N" -i "$I" -w "$3"
    shopt -s nullglob
    local V=( "$3"/violation-* )
    echo " [$1] violations: ${#V[@]}"
    eval "$4=${#V[@]}"
}

run_one "WITH-trainer" "$CAMP/template.asm"                  "$CAMP/out-residue-withtrain" AV
run_one "NO-trainer"   "$CAMP/template-residue-notrain.asm"  "$CAMP/out-residue-notrain"   BV

echo "=================================================================="
echo " RESULT (N=$N each):  WITH-trainer=$AV   NO-trainer=$BV"
if [ "$AV" -gt 0 ] && [ "$BV" -eq 0 ]; then
    echo " => trainer NECESSARY  ==>  GENUINE cross-guest BTB residue"
elif [ "$AV" -gt 0 ] && [ "$BV" -gt 0 ]; then
    echo " => trainer irrelevant ==>  generic transient, NOT residue"
elif [ "$AV" -eq 0 ]; then
    echo " => WITH-trainer didn't even leak at N=$N — increase N (original hit at TC 585)."
fi
echo "=================================================================="
