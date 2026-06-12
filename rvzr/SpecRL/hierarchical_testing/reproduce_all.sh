#!/usr/bin/env bash
#
# Batch-reproduce Revizor violations.
#
# For every violation-* folder under hierarchical_testing it runs:
#   ./revizor.py reproduce -s base.json \
#       -c <folder>/reproduce.yaml \
#       -t <folder>/program.asm \
#       -i <folder>/input*.bin
#
# Output of each run is saved to <folder>/reproduce_result.txt, and a summary
# table (per-folder exit code + reproduced? + #violations) is printed at the end
# and written to ./reproduce_summary.tsv (in the hierarchical_testing dir).
#
# Usage:
#   ./reproduce_all.sh                 # default: only violation-260610-*  (today's div batch)
#   ./reproduce_all.sh 'violation-*'   # ALL violation folders (slow: ~2min each)
#   ./reproduce_all.sh 'violation-260609-05*'   # any glob you like
#
# If the executor needs root (it usually does — it touches /sys/rvzr_executor),
# run the whole thing under sudo:   sudo ./reproduce_all.sh
# or set SUDO=sudo below.
#
# ---------------------------------------------------------------------------

set -u  # NOT -e: we want to continue past individual failures.

# --- paths ------------------------------------------------------------------
REVIZOR_DIR="/home/hz25d/sca-fuzzer"                                  # where revizor.py + base.json live
VIOL_DIR="/home/hz25d/sca-fuzzer/rvzr/SpecRL/hierarchical_testing"    # where the violation-* folders live
BASE_CFG="base.json"                                                  # passed to -s, resolved from REVIZOR_DIR

# --- knobs ------------------------------------------------------------------
PATTERN="${1:-violation-260610-*}"   # which folders to run (glob, relative to VIOL_DIR)
SUDO="${SUDO:-}"                      # set SUDO=sudo to run each reproduce as root
TIMEOUT="${TIMEOUT:-600}"            # per-run timeout (seconds); 0 = no timeout

# SSB_OFF=1 (default): force the executor SSB-vulnerable during reproduce, i.e.
# replicate the TRAINING condition. Without this, reproduce runs with the rvzr
# default x86_executor_enable_ssbp_patch=True (SSBD ON), which MITIGATES
# Spectre-v4 and makes genuine v4 violations look like "false positives" / not
# reproducible. With SSB_OFF=1 the script appends
#     x86_executor_enable_ssbp_patch: false
# to a temp copy of each reproduce.yaml; executor._set_vendor_specific_features()
# then writes "0" to /sys/rvzr_executor/enable_ssbp_patch on init.
#   SSB_OFF=1  -> reproduce with SSBD OFF (correct test for v4)         [default]
#   SSB_OFF=0  -> reproduce exactly as-is (SSBD ON, rvzr default)
# To compare, run once each way and diff reproduce_summary.tsv.
SSB_OFF="${SSB_OFF:-1}"

# CONTRACT (optional): override the model's execution clause to CLASSIFY a
# confirmed violation. If a known speculation contract "explains" it (the
# violation DISAPPEARS), it's that known class; if it SURVIVES even the
# strongest known contract, it's a NEW-attack candidate.
#   CONTRACT=bpas            -> Spectre-v4 / store-bypass model  (does bpas explain it?)
#   CONTRACT=vspec-all-div   -> variable-latency DIV model
#   CONTRACT=cond-bpas       -> v1+v4 combined
#   (unset, default)         -> use rvzr default contract (delayed-exception-handling)
# Appended to the temp config as a YAML list: `contract_execution_clause: [<v>]`.
CONTRACT="${CONTRACT:-}"

_sfx="$([ "$SSB_OFF" = 1 ] && echo _ssboff)$([ -n "$CONTRACT" ] && echo "_${CONTRACT//[^a-zA-Z0-9]/-}")"
RESULT_NAME="reproduce_result${_sfx}.txt"
SUMMARY="${VIOL_DIR}/reproduce_summary${_sfx}.tsv"

# ---------------------------------------------------------------------------
cd "$REVIZOR_DIR" || { echo "ERROR: cannot cd to $REVIZOR_DIR"; exit 1; }

if [[ ! -x ./revizor.py && ! -f ./revizor.py ]]; then
    echo "ERROR: ./revizor.py not found in $REVIZOR_DIR"; exit 1
fi
if [[ ! -f "$BASE_CFG" ]]; then
    echo "ERROR: $BASE_CFG not found in $REVIZOR_DIR"; exit 1
fi

# preflight: make sure revizor.py can actually import its deps in THIS
# environment. The #1 gotcha is running under `sudo`, which drops the conda/
# venv and yields `ModuleNotFoundError: No module named 'numpy'` — every run
# then crashes at import and silently writes 24 garbage results. The executor
# sysfs is world-writable, so root is NOT needed; run without sudo.
if ! $SUDO python3 -c "import numpy" >/dev/null 2>&1; then
    echo "ERROR: python3 in this environment can't import numpy"
    echo "       (\$SUDO='${SUDO:-<none>}', python3=$($SUDO which python3 2>/dev/null))"
    echo "  Most likely you're under sudo, which loses your conda/venv."
    echo "  Fix: run WITHOUT sudo (executor sysfs is world-writable, root not needed),"
    echo "       or point SUDO at an env-preserving wrapper that keeps numpy."
    exit 1
fi

# build timeout prefix
TO_PREFIX=()
if [[ "$TIMEOUT" != "0" ]]; then
    TO_PREFIX=(timeout --signal=INT "$TIMEOUT")
fi

# collect matching folders (sorted)
shopt -s nullglob
folders=( "$VIOL_DIR"/$PATTERN/ )
shopt -u nullglob
if [[ ${#folders[@]} -eq 0 ]]; then
    echo "No folders matched: $VIOL_DIR/$PATTERN"; exit 1
fi

printf 'folder\texit_code\treproduced\tn_violations\n' > "$SUMMARY"
total=${#folders[@]}
echo "Running 'revizor.py reproduce' on $total folder(s) matching '$PATTERN'"
echo "Working dir: $REVIZOR_DIR   |   per-run log: <folder>/$RESULT_NAME"
echo "Summary -> $SUMMARY"
echo "============================================================"

i=0
for dir in "${folders[@]}"; do
    dir="${dir%/}"
    i=$((i+1))
    name="$(basename "$dir")"
    asm="$dir/program.asm"
    cfg="$dir/reproduce.yaml"
    log="$dir/$RESULT_NAME"

    # expand the input files for this folder
    inputs=( "$dir"/input*.bin )

    printf '[%d/%d] %s ... ' "$i" "$total" "$name"

    # sanity checks
    if [[ ! -f "$asm" || ! -f "$cfg" || ${#inputs[@]} -eq 0 ]]; then
        echo "SKIP (missing program.asm / reproduce.yaml / input*.bin)"
        printf '%s\t%s\t%s\t%s\n' "$name" "SKIP" "-" "-" >> "$SUMMARY"
        continue
    fi

    # effective config: optionally force SSBD OFF and/or override the contract
    eff_cfg="$cfg"
    tmp_cfg=""
    if [[ "$SSB_OFF" = "1" || -n "$CONTRACT" ]]; then
        tmp_cfg="$(mktemp)"
        cat "$cfg" > "$tmp_cfg"
        [[ "$SSB_OFF" = "1" ]] && printf '\nx86_executor_enable_ssbp_patch: false\n' >> "$tmp_cfg"
        [[ -n "$CONTRACT" ]]   && printf '\ncontract_execution_clause: [%s]\n' "$CONTRACT" >> "$tmp_cfg"
        eff_cfg="$tmp_cfg"
    fi

    # run it; capture stdout+stderr to the per-folder log
    $SUDO "${TO_PREFIX[@]}" ./revizor.py reproduce \
        -s "$BASE_CFG" \
        -c "$eff_cfg" \
        -t "$asm" \
        -i "${inputs[@]}" \
        > "$log" 2>&1
    rc=$?
    [[ -n "$tmp_cfg" ]] && rm -f "$tmp_cfg"

    # heuristics to read the result out of the log (Revizor's wording can vary;
    # adjust the greps here if your build prints something different).
    nviol="$(grep -aoiE 'violations?[^0-9]*([0-9]+)' "$log" | grep -aoE '[0-9]+' | tail -1)"
    nviol="${nviol:-?}"
    if grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then
        repro="NO"
    elif grep -aqiE 'violation' "$log"; then
        repro="YES"
    else
        repro="?"
    fi

    if [[ $rc -eq 124 ]]; then
        echo "TIMEOUT (${TIMEOUT}s)"
    else
        echo "done (rc=$rc, reproduced=$repro, n_violations=$nviol)"
    fi
    printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$repro" "$nviol" >> "$SUMMARY"
done

echo "============================================================"
echo "Summary (also in $SUMMARY):"
column -t -s $'\t' "$SUMMARY"
