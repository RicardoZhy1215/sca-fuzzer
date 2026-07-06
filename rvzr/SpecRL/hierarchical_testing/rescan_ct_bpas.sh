#!/usr/bin/env bash
# Re-classify ct+bpas reproduce runs from the ALREADY-WRITTEN logs.
# No executor, no re-run -- just reads <folder>/reproduce_ct_bpas.log.
# Distinguishes real verdicts from the slow-path taint-tracker crash.
set -u
VIOL_DIR="/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing"
PATTERN="${1:-violation-26061[234]-*}"
OUT="${VIOL_DIR}/reproduce_ct_bpas.tsv"

shopt -s nullglob; folders=( "$VIOL_DIR"/$PATTERN/ ); shopt -u nullglob
printf 'folder\tverdict\tn_violations\n' > "$OUT"
no=0; yes=0; err=0; tmo=0; miss=0
for dir in "${folders[@]}"; do
    dir="${dir%/}"; name="$(basename "$dir")"; log="$dir/reproduce_ct_bpas.log"
    [ -f "$log" ] || { printf '%s\tNO_LOG\t-\n' "$name">>"$OUT"; miss=$((miss+1)); continue; }
    nviol="$(grep -aoiE 'violations?: *([0-9]+)' "$log" | grep -aoE '[0-9]+' | tail -1)"
    if   grep -aqiE 'KeyboardInterrupt' "$log"; then v=TIMEOUT; tmo=$((tmo+1))
    elif grep -aqiE 'no more checkpoints|Traceback|AssertionError' "$log"; then v=ERROR; err=$((err+1))
    elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then v=NO; no=$((no+1))
    elif grep -aqiE 'violation' "$log"; then v=YES; yes=$((yes+1))
    else v="?"; fi
    printf '%s\t%s\t%s\n' "$name" "$v" "${nviol:-?}" >> "$OUT"
done
echo "=== ct+bpas verdict (from logs) ==="
echo "  NO (V4, bpas explains)      : $no"
echo "  YES (survived all filters)  : $yes"
echo "  ERROR (slow-path taint crash): $err"
echo "  TIMEOUT                     : $tmo"
echo "  missing log                 : $miss"
echo "Detail -> $OUT"
echo
echo "ERROR/TIMEOUT folders (fast-path flagged under bpas, slow path crashed):"
awk -F'\t' '$2=="ERROR"||$2=="TIMEOUT"{print "  "$1"  ("$2")"}' "$OUT"
