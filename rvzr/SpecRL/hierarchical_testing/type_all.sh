#!/usr/bin/env bash
# Per-violation TYPE classification via a contract ladder (executor SSBD OFF,
# matching the RL generation condition). For each violation we replay it under
# progressively different contracts; the FIRST contract that makes the violation
# disappear names the leak type. Short-circuits (stops at first explainer).
#
# Ladder (label <- the contract that explains it):
#   1. delayed-exception-handling : reproducibility gate. If it does NOT
#      reproduce here either -> FLAKY (noise / unfiltered false positive).
#   2. bpas                       : V4        (speculative store bypass)
#   3. [cond, bpas]               : V1+V4     (also needs branch mispred.)
#   4. vspec-all-div              : VARLAT-DIV (data-dependent divider latency)
#   5. vspec-all-memory-assists   : ASSIST    (microcode assist replay)
#   6. vspec-all-memory-faults    : FAULT     (faulting-load speculation)
#   else                          : NOVEL     (survives every tested contract)
#
# Output: type_summary.tsv  (folder \t repro \t explained_by \t type)
set -u
REVIZOR_DIR="/home/mluo/sca-fuzzer"
VIOL_DIR="/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing"
BASE_CFG="base.json"
PATTERN="${1:-violation-26061[234]-*}"
TIMEOUT="${TIMEOUT:-300}"
OUT="${VIOL_DIR}/type_summary.tsv"
cd "$REVIZOR_DIR" || exit 1

# replay $1=folder under contract list-literal $2 (e.g. "[bpas]"; ""=default).
# echoes YES (still a violation) / NO (explained) / ?(unknown)
replay() {
    local dir="$1" clause="$2" tag="$3"
    local tmp; tmp="$(mktemp)"; cat "$dir/reproduce.yaml" > "$tmp"
    printf '\nx86_executor_enable_ssbp_patch: false\n' >> "$tmp"
    [ -n "$clause" ] && printf 'contract_execution_clause: %s\n' "$clause" >> "$tmp"
    local inputs=( "$dir"/input*.bin )
    local log="$dir/type_${tag}.log"
    timeout --signal=INT "$TIMEOUT" ./revizor.py reproduce -s "$BASE_CFG" \
        -c "$tmp" -t "$dir/program.asm" -i "${inputs[@]}" > "$log" 2>&1
    rm -f "$tmp"
    if   grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then echo NO
    elif grep -aqiE 'violation' "$log"; then echo YES
    else echo "?"; fi
}

shopt -s nullglob; folders=( "$VIOL_DIR"/$PATTERN/ ); shopt -u nullglob
total=${#folders[@]}
printf 'folder\trepro\texplained_by\ttype\n' > "$OUT"
echo "Typing $total folders (SSBD OFF, short-circuit ladder). -> $OUT"

i=0
for dir in "${folders[@]}"; do
    dir="${dir%/}"; i=$((i+1)); name="$(basename "$dir")"
    [ -f "$dir/program.asm" ] || { printf '%s\tSKIP\t-\tSKIP\n' "$name">>"$OUT"; continue; }

    # gate: does it reproduce under the generation contract at all?
    if [ "$(replay "$dir" "" delayedexc)" = "NO" ]; then
        printf '%s\tNO\t-\tFLAKY\n' "$name" >>"$OUT"; echo "[$i/$total] $name -> FLAKY"; continue
    fi
    typ=NOVEL; exby="-"
    for pair in "[bpas]:V4:bpas" \
                "[cond, bpas]:V1+V4:cond-bpas" \
                "[vspec-all-div]:VARLAT-DIV:vspec-all-div" \
                "[vspec-all-memory-assists]:ASSIST:vspec-all-memory-assists" \
                "[vspec-all-memory-faults]:FAULT:vspec-all-memory-faults"; do
        clause="${pair%%:*}"; rest="${pair#*:}"; label="${rest%%:*}"; tag="${rest#*:}"
        if [ "$(replay "$dir" "$clause" "$tag")" = "NO" ]; then typ="$label"; exby="$tag"; break; fi
    done
    printf '%s\tYES\t%s\t%s\n' "$name" "$exby" "$typ" >>"$OUT"
    echo "[$i/$total] $name -> $typ (explained_by=$exby)"
done
echo "=== TYPE histogram ==="; tail -n +2 "$OUT" | cut -f4 | sort | uniq -c | sort -rn
