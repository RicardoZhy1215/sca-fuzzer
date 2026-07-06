#!/usr/bin/env bash
# Fresh full replay of every post-6/12 violation under the SAME contract that
# generated them (big_fuzz.yaml = ct + bpas + P+P + filters + inputs_per_class:2),
# executor SSBD forced OFF (matching the RL env). Counts how many reproduce.
#
# Verdict per folder:
#   NO       fast path found no divergence -> bpas explains it      -> V4
#   YES      survived ALL slow-path filters -> beyond-V4 candidate
#   ERROR    bpas taint-tracker crashed in boosting ("no more checkpoints")
#   TIMEOUT  slow-path deepcopy too slow within $TIMEOUT
#
# Usage:  ./run_ct_bpas.sh                      # all post-6/12, fresh
#         ./run_ct_bpas.sh 'violation-260613-*' # a subset
set -u

REVIZOR_DIR="/home/mluo/sca-fuzzer"
VIOL_DIR="/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing"
BASE_CFG="base.json"
GEN_CFG="${VIOL_DIR}/big_fuzz.yaml"
PATTERN="${1:-violation-26061[234]-*}"
TIMEOUT="${TIMEOUT:-180}"          # 3 min/folder; stuck taint cases -> TIMEOUT
SSBD_OFF="${SSBD_OFF:-1}"
SUMMARY="${VIOL_DIR}/run_ct_bpas.tsv"

cd "$REVIZOR_DIR" || { echo "cannot cd to $REVIZOR_DIR"; exit 1; }
[ -f "$GEN_CFG" ] || { echo "missing $GEN_CFG"; exit 1; }

shopt -s nullglob; folders=( "$VIOL_DIR"/$PATTERN/ ); shopt -u nullglob
total=${#folders[@]}
[ "$total" -eq 0 ] && { echo "no folders match: $PATTERN"; exit 1; }

printf 'folder\tverdict\tn_violations\n' > "$SUMMARY"
echo "Replaying $total folders under ct+bpas (big_fuzz.yaml, SSBD_OFF=$SSBD_OFF, TIMEOUT=${TIMEOUT}s)"
echo "Summary -> $SUMMARY"
echo "============================================================"

no=0; yes=0; err=0; tmo=0; oth=0; i=0
for dir in "${folders[@]}"; do
    dir="${dir%/}"; i=$((i+1)); name="$(basename "$dir")"
    asm="$dir/program.asm"; inputs=( "$dir"/input*.bin )
    if [ ! -f "$asm" ] || [ ${#inputs[@]} -eq 0 ]; then
        printf '%s\tSKIP\t-\n' "$name" >>"$SUMMARY"; echo "[$i/$total] $name SKIP"; continue
    fi

    cfg="$(mktemp)"
    grep -v '^[[:space:]]*x86_executor_enable_ssbp_patch:' "$GEN_CFG" > "$cfg"
    [ -f "$dir/reproduce.yaml" ] && grep -E '^[a-zA-Z].*seed' "$dir/reproduce.yaml" >> "$cfg"
    [ "$SSBD_OFF" = "1" ] && echo "x86_executor_enable_ssbp_patch: false" >> "$cfg" \
                          || echo "x86_executor_enable_ssbp_patch: true"  >> "$cfg"

    log="$dir/run_ct_bpas.log"
    timeout --signal=INT "$TIMEOUT" ./revizor.py reproduce \
        -s "$BASE_CFG" -c "$cfg" -t "$asm" -i "${inputs[@]}" > "$log" 2>&1
    rc=$?
    rm -f "$cfg"

    nviol="$(grep -aoiE 'violations?: *([0-9]+)' "$log" | grep -aoE '[0-9]+' | tail -1)"
    if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then v=TIMEOUT; tmo=$((tmo+1))
    elif grep -aqiE 'no more checkpoints|Traceback|AssertionError' "$log"; then v=ERROR; err=$((err+1))
    elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then v=NO; no=$((no+1))
    elif grep -aqiE 'violation' "$log"; then v=YES; yes=$((yes+1))
    else v="?"; oth=$((oth+1)); fi
    printf '%s\t%s\t%s\n' "$name" "$v" "${nviol:-?}" >>"$SUMMARY"
    echo "[$i/$total] $name  -> $v"
done

echo "============================================================"
echo "DONE.  NO(V4)=$no  YES(beyond-V4)=$yes  ERROR=$err  TIMEOUT=$tmo  other=$oth  / total=$total"
echo "Detail -> $SUMMARY"
