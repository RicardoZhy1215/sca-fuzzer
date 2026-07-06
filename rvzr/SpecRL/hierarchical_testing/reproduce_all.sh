#!/usr/bin/env bash
#
# Batch-reproduce Revizor violations.
#
# For every violation-* folder under hierarchical_testing it runs:
#   ./revizor.py reproduce -s base.json \
#       -c <folder>/reproduce.yaml (+ overrides) \
#       -t <folder>/program.asm \
#       -i <folder>/input*.bin
#
# Output of each run is saved to <folder>/reproduce_result*.txt, and a summary
# table (per-folder exit code + reproduced? + #violations) is printed at the end
# and written to ./reproduce_summary*.tsv (in the hierarchical_testing dir).
#
# Usage:
#   ./reproduce_all.sh                 # default glob: violation-260610-*
#   ./reproduce_all.sh 'violation-*'   # ALL violation folders
#   ./reproduce_all.sh 'violation-260701-*'
#
# ---------------------------------------------------------------------------
# FAITHFUL reproduction (same contract that FOUND these violations = big_fuzz.yaml):
#
#   CONTRACT='cond, bpas' OBS=ct ./reproduce_all.sh 'violation-260701-*'
#
# Why you MUST pass these: each per-folder reproduce.yaml only stores a seed and
# inputs_per_class:1, and (falsely) implies "default config". The real generation
# used big_fuzz.yaml = `contract_observation_clause: ct` + `contract_execution_clause:
# [cond, bpas]`. Replaying with the rvzr DEFAULT contract (delayed-exception-handling
# + loads+stores+pc) is NOT the same experiment.
#
# CLASSIFY a confirmed violation (does a known contract explain it? if it DISAPPEARS,
# that contract explains it; if it SURVIVES the strongest known contract -> new-attack
# candidate):
#   CONTRACT=bpas          OBS=ct ./reproduce_all.sh ...   # Spectre-v4 / store-bypass only
#   CONTRACT='cond, bpas'  OBS=ct ./reproduce_all.sh ...   # v1+v4 combined
#   CONTRACT=vspec-all-div OBS=ct ./reproduce_all.sh ...   # variable-latency DIV
#
#   IMPORTANT: pass the clause LIST BODY, comma-separated. 'cond-bpas' is NOT a valid
#   execution clause (see config.py allowed list); the combined cond+bpas speculator is
#   selected only when BOTH 'cond' and 'bpas' appear in the list -> pass 'cond, bpas'.
# ---------------------------------------------------------------------------

set -u  # NOT -e: we want to continue past individual failures.

# --- paths ------------------------------------------------------------------
REVIZOR_DIR="/home/mluo/sca-fuzzer"                                  # where revizor.py + base.json live
VIOL_DIR="/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing"    # where the violation-* folders live
BASE_CFG="base.json"                                                 # passed to -s, resolved from REVIZOR_DIR

# --- knobs ------------------------------------------------------------------
PATTERN="${1:-violation-260610-*}"   # which folders to run (glob, relative to VIOL_DIR)
SUDO="${SUDO:-}"                      # normally EMPTY: /sys/rvzr_executor is world-writable, root NOT needed.
                                      # (running under sudo drops conda/venv -> numpy import fails; see preflight.)
TIMEOUT="${TIMEOUT:-600}"            # per-run timeout (seconds); 0 = no timeout

# SSB_OFF=1 (default): force the executor SSB-vulnerable during reproduce, i.e.
# replicate the TRAINING condition. Without this, reproduce runs with the rvzr
# default SSBD ON, which MITIGATES Spectre-v4 and makes genuine v4 violations
# look like "false positives" / not reproducible.
#   SSB_OFF=1  -> reproduce with SSBD OFF (correct test for store-bypass)  [default]
#   SSB_OFF=0  -> keep SSBD ON (rvzr default; store-bypass mitigated)
SSB_OFF="${SSB_OFF:-1}"

# CONTRACT (optional): execution-clause LIST BODY, comma-separated. Empty = rvzr
# default (delayed-exception-handling). Examples: 'cond, bpas' | 'bpas' | 'seq'.
CONTRACT="${CONTRACT:-}"

# OBS (optional): observation clause. Empty = folder default (loads+stores+pc).
# For faithful reproduction of these violations use OBS=ct (big_fuzz.yaml used ct).
OBS="${OBS:-}"

# IPC = inputs_per_class. Default 1 = NO boosting / NO taint tracking.
#   Rationale: with IPC>1 the model runs trace_test_case_with_taints, which does a
#   copy.deepcopy of the taint state on EVERY speculative rollback (taint_tracker.py
#   checkpoint/rollback). On store-heavy code with bpas that is thousands of deepcopies
#   -> the slow path appears to "hang" (it is just crawling). The saved input*.bin
#   already contain the violating inputs, so IPC=1 both reproduces AND stays fast.
#   Only raise IPC if a violation shows NO and you suspect it needed boosted inputs.
IPC="${IPC:-1}"

_c="${CONTRACT//[^a-zA-Z0-9]/}"; _o="${OBS//[^a-zA-Z0-9]/}"
_sfx="$([ "$SSB_OFF" = 1 ] && echo _ssboff)${_c:+_$_c}${_o:+_obs$_o}$([ "$IPC" != 1 ] && echo "_ipc$IPC")"
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

# preflight: make sure revizor.py can import its deps in THIS environment. The #1
# gotcha is running under `sudo`, which drops the conda/venv and yields
# `ModuleNotFoundError: No module named 'numpy'`. The executor sysfs is
# world-writable, so root is NOT needed; run without sudo.
if ! $SUDO python3 -c "import numpy" >/dev/null 2>&1; then
    echo "ERROR: python3 in this environment can't import numpy"
    echo "       (\$SUDO='${SUDO:-<none>}', python3=$($SUDO which python3 2>/dev/null))"
    echo "  Most likely you're under sudo, which loses your conda/venv."
    echo "  Fix: run WITHOUT sudo (executor sysfs is world-writable, root not needed)."
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
echo "contract=[${CONTRACT:-rvzr-default}]  obs=${OBS:-folder-default}  inputs_per_class=$IPC  ssb_off=$SSB_OFF"
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

    # effective config = folder's reproduce.yaml, with every key we're about to set
    # stripped first (so we never emit duplicate YAML keys), then re-appended.
    tmp_cfg="$(mktemp)"
    grep -vE '^[[:space:]]*(x86_executor_enable_ssbp_patch|inputs_per_class|contract_observation_clause|contract_execution_clause):|^[[:space:]]*-[[:space:]]*(seq|bpas|cond|cond-bpas|seq-assist|no_speculation|conditional_br_misprediction|delayed-exception-handling|div-zero|div-overflow|meltdown|fault-skip|noncanonical|nullinj-fault|nullinj-assist|vspec-[a-z-]+)[[:space:]]*$' "$cfg" > "$tmp_cfg"

    # pin inputs_per_class (default 1 = no boosting/taint-tracking; avoids the
    # per-rollback copy.deepcopy blowup that makes the slow path look hung).
    printf 'inputs_per_class: %s\n' "$IPC" >> "$tmp_cfg"
    # force SSBD off so the CPU can actually store-bypass (training condition).
    [[ "$SSB_OFF" = "1" ]] && printf 'x86_executor_enable_ssbp_patch: false\n' >> "$tmp_cfg"
    # optional contract overrides (faithful repro / classification).
    [[ -n "$CONTRACT" ]] && printf 'contract_execution_clause: [%s]\n' "$CONTRACT" >> "$tmp_cfg"
    [[ -n "$OBS" ]]      && printf 'contract_observation_clause: %s\n'  "$OBS"      >> "$tmp_cfg"
    eff_cfg="$tmp_cfg"

    # run it; capture stdout+stderr to the per-folder log
    $SUDO "${TO_PREFIX[@]}" ./revizor.py reproduce \
        -s "$BASE_CFG" \
        -c "$eff_cfg" \
        -t "$asm" \
        -i "${inputs[@]}" \
        > "$log" 2>&1
    rc=$?
    rm -f "$tmp_cfg"

    # heuristics to read the result out of the log. Order matters: check crash/timeout
    # BEFORE the generic 'violation' match, so a traceback that merely contains the word
    # "violation" is not misread as a successful reproduction.
    nviol="$(grep -aoiE 'violations?[^0-9]*([0-9]+)' "$log" | grep -aoE '[0-9]+' | tail -1)"
    nviol="${nviol:-?}"
    if [[ $rc -eq 124 ]] || grep -aqiE 'KeyboardInterrupt' "$log"; then
        repro="TIMEOUT"
    elif grep -aqiE 'no more checkpoints|Traceback|AssertionError|Exception' "$log"; then
        repro="ERROR"
    elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then
        repro="NO"
    elif grep -aqiE 'violation' "$log"; then
        repro="YES"
    else
        repro="?"
    fi

    if [[ "$repro" == "TIMEOUT" ]]; then
        echo "TIMEOUT (${TIMEOUT}s)"
    else
        echo "done (rc=$rc, reproduced=$repro, n_violations=$nviol)"
    fi
    printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$repro" "$nviol" >> "$SUMMARY"
done

echo "============================================================"
echo "Summary (also in $SUMMARY):"
column -t -s $'\t' "$SUMMARY"
