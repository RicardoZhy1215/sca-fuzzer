#!/usr/bin/env bash
# Replay every violation under the SAME config that generated them
# (big_fuzz.yaml = ct + bpas + P+P + filters + inputs_per_class:2) and count
# how many still reproduce as a violation.
#
# Why big_fuzz.yaml instead of reproduce.yaml: the per-folder reproduce.yaml
# only carries a seed and (falsely) says "default config"; the real generation
# used big_fuzz.yaml, which also sets the contract, executor mode, the three
# filters, and inputs_per_class. Replaying with that config is the faithful test.
#
#   contract_observation_clause: ct      (from big_fuzz.yaml)
#   contract_execution_clause:  [bpas]   (from big_fuzz.yaml)
#
# A violation that STILL reproduces survives bpas -> NOT plain Spectre-v4.
# One that does NOT reproduce -> bpas explains it -> it is V4.
#
# Usage:
#   ./reproduce_ct_bpas.sh                       # default: post-6/12 batch
#   ./reproduce_ct_bpas.sh 'violation-260613-*'  # any glob (relative to VIOL_DIR)
#
# SSBD: big_fuzz.yaml sets the patch ON, but the RL env force-disables it at
# runtime (so the hardware can actually do store-bypass). We mirror that by
# overriding the patch OFF here. Set SSBD_OFF=0 to keep it ON (then nothing
# leaks and everything "doesn't reproduce" for an uninteresting reason).
set -u

REVIZOR_DIR="/home/mluo/sca-fuzzer"
VIOL_DIR="/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing"
BASE_CFG="base.json"                              # ISA spec (-s)
GEN_CFG="${VIOL_DIR}/big_fuzz.yaml"               # the generation config (ct+bpas)
PATTERN="${1:-violation-26061[234]-*}"            # which folders (post-6/12 by default)
TIMEOUT="${TIMEOUT:-300}"                          # per-replay timeout (s)
SSBD_OFF="${SSBD_OFF:-1}"                          # 1 = SSBD off (default), 0 = on
SUMMARY="${VIOL_DIR}/reproduce_ct_bpas.tsv"

cd "$REVIZOR_DIR" || { echo "cannot cd to $REVIZOR_DIR"; exit 1; }
[ -f "$GEN_CFG" ] || { echo "missing generation config: $GEN_CFG"; exit 1; }

shopt -s nullglob
folders=( "$VIOL_DIR"/$PATTERN/ )
shopt -u nullglob
total=${#folders[@]}
[ "$total" -eq 0 ] && { echo "no folders match: $PATTERN"; exit 1; }

printf 'folder\treproduced\tn_violations\n' > "$SUMMARY"
echo "Replaying $total folders under big_fuzz.yaml (ct+bpas, SSBD_OFF=$SSBD_OFF). -> $SUMMARY"
echo "========================================================================"

repro=0; notrepro=0; other=0; err=0; tmo=0; i=0
for dir in "${folders[@]}"; do
    dir="${dir%/}"; i=$((i+1)); name="$(basename "$dir")"
    asm="$dir/program.asm"; inputs=( "$dir"/input*.bin )
    if [ ! -f "$asm" ] || [ ${#inputs[@]} -eq 0 ]; then
        printf '%s\tSKIP\t-\n' "$name" >> "$SUMMARY"; echo "[$i/$total] $name SKIP"; continue
    fi

    # effective config = big_fuzz.yaml, with the SSBD patch line stripped and
    # re-appended to the desired value, plus the per-folder seed (harmless;
    # -i overrides input generation anyway).
    cfg="$(mktemp)"
    grep -v '^[[:space:]]*x86_executor_enable_ssbp_patch:' "$GEN_CFG" > "$cfg"
    [ -f "$dir/reproduce.yaml" ] && grep -E '^[a-zA-Z].*seed' "$dir/reproduce.yaml" >> "$cfg"
    if [ "$SSBD_OFF" = "1" ]; then
        echo "x86_executor_enable_ssbp_patch: false" >> "$cfg"
    else
        echo "x86_executor_enable_ssbp_patch: true"  >> "$cfg"
    fi

    log="$dir/reproduce_ct_bpas.log"
    timeout --signal=INT "$TIMEOUT" ./revizor.py reproduce \
        -s "$BASE_CFG" -c "$cfg" -t "$asm" -i "${inputs[@]}" > "$log" 2>&1
    rc=$?
    rm -f "$cfg"

    nviol="$(grep -aoiE 'violations?: *([0-9]+)' "$log" | grep -aoE '[0-9]+' | tail -1)"
    # Crashes/timeouts must NOT be read as violations. The slow-path boosting
    # stage can crash the bpas speculator's taint tracker ("no more
    # checkpoints"); such a run had a fast-path divergence but could not be
    # confirmed. Label these distinctly instead of calling them YES.
    if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then
        res=TIMEOUT; tmo=$((tmo+1))
    elif grep -aqiE 'no more checkpoints|Traceback|AssertionError|Exception' "$log"; then
        res=ERROR; err=$((err+1))
    elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then
        res=NO;  notrepro=$((notrepro+1))
    elif grep -aqiE 'violation' "$log"; then
        res=YES; repro=$((repro+1))
    else
        res="?"; other=$((other+1))
    fi
    printf '%s\t%s\t%s\n' "$name" "$res" "${nviol:-?}" >> "$SUMMARY"
    echo "[$i/$total] $name  reproduced=$res  n_violations=${nviol:-?}"
done

echo "========================================================================"
echo "ct+bpas replay done.  reproduced(YES)=$repro  not-reproduced(NO)=$notrepro  ERROR=$err  TIMEOUT=$tmo  unknown=$other  / total=$total"
echo "  NO  = bpas explains it (fast path clean)            -> V4"
echo "  YES = survived all slow-path filters                -> beyond-V4 candidate"
echo "  ERROR/TIMEOUT = fast-path divergence, slow path crashed (bpas taint bug) -> needs manual check"
echo "Summary: $SUMMARY"
