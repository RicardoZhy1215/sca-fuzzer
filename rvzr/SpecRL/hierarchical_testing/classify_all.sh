#!/usr/bin/env bash
# Classify post-6/12 violations by reproducing each under two contracts
# (executor SSBD forced OFF, matching the RL generation condition):
#   A) delayed-exception-handling (Revizor default)  -> does it reproduce at all?
#   B) bpas (Spectre-v4 / store-bypass model)         -> does v4 explain it?
#
# Class:
#   FLAKY      reproduced in NEITHER         -> noise / unfiltered false positive
#   V4         reproduced in A, NOT in B     -> store-bypass, explained by bpas
#   BEYOND-V4  reproduced in BOTH A and B    -> survives bpas (interesting)
#   ODD        reproduced in B only          -> shouldn't happen; flag it
#
# Output: classify_summary.tsv  +  per-folder logs (reuse reproduce_all naming).
set -u
REVIZOR_DIR="/home/mluo/sca-fuzzer"
VIOL_DIR="/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing"
BASE_CFG="base.json"
PATTERN="${1:-violation-2606[12-14]-*}"
TIMEOUT="${TIMEOUT:-300}"
OUT="${VIOL_DIR}/classify_summary.tsv"

cd "$REVIZOR_DIR" || exit 1
shopt -s nullglob
folders=( "$VIOL_DIR"/$PATTERN/ )
shopt -u nullglob
total=${#folders[@]}
printf 'folder\trepro_delayedexc\trepro_bpas\tclass\n' > "$OUT"
echo "Classifying $total folders (SSBD OFF). Output -> $OUT"

run_one() { # $1=folder $2=contract(empty=default)  -> echo YES/NO
    local dir="$1" contract="$2"
    local cfg tmp; cfg="$dir/reproduce.yaml"
    tmp="$(mktemp)"; cat "$cfg" > "$tmp"
    printf '\nx86_executor_enable_ssbp_patch: false\n' >> "$tmp"
    [ -n "$contract" ] && printf 'contract_execution_clause: [%s]\n' "$contract" >> "$tmp"
    local inputs=( "$dir"/input*.bin )
    local log; log="$dir/classify_$( [ -n "$contract" ] && echo "$contract" || echo "default" ).log"
    timeout --signal=INT "$TIMEOUT" ./revizor.py reproduce -s "$BASE_CFG" \
        -c "$tmp" -t "$dir/program.asm" -i "${inputs[@]}" > "$log" 2>&1
    rm -f "$tmp"
    if grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then echo NO
    elif grep -aqiE 'violation' "$log"; then echo YES; else echo "?"; fi
}

i=0
for dir in "${folders[@]}"; do
    dir="${dir%/}"; i=$((i+1)); name="$(basename "$dir")"
    [ -f "$dir/program.asm" ] || { printf '%s\tSKIP\tSKIP\tSKIP\n' "$name" >>"$OUT"; continue; }
    a=$(run_one "$dir" "")
    b=$(run_one "$dir" "bpas")
    case "$a/$b" in
        NO/NO)   cls=FLAKY ;;
        YES/NO)  cls=V4 ;;
        YES/YES) cls=BEYOND-V4 ;;
        NO/YES)  cls=ODD ;;
        *)       cls="A=$a,B=$b" ;;
    esac
    printf '%s\t%s\t%s\t%s\n' "$name" "$a" "$b" "$cls" >> "$OUT"
    echo "[$i/$total] $name  delayedexc=$a bpas=$b -> $cls"
done
echo "=== class counts ==="
tail -n +2 "$OUT" | cut -f4 | sort | uniq -c | sort -rn
