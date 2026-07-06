#!/usr/bin/env bash
# Reproduce ONE violation folder faithfully.
#
# Why not the per-folder reproduce.yaml: it only carries a seed and (falsely)
# says "default config". The real generation used big_fuzz.yaml, which sets the
# contract (ct + [cond,bpas]), executor mode (P+P) and the filters. We replay
# with THAT, but force the SSBP patch OFF (the RL env force-disables SSBD at
# runtime; with it ON the CPU won't store-bypass and nothing leaks).
#
# Usage:
#   ./reproduce_one.sh <violation-folder-name> [num_inputs]
# Examples:
#   ./reproduce_one.sh violation-260701-234841
#   CLAUSE='bpas'      ./reproduce_one.sh violation-260701-234841   # is it explained by v4 alone?
#   CLAUSE='seq'       ./reproduce_one.sh violation-260701-234841   # sanity: no-spec contract
#   SSBD_OFF=0         ./reproduce_one.sh violation-260701-234841   # keep SSBD patch ON (should NOT leak)
#   TIMEOUT=900 NI=100 ./reproduce_one.sh violation-260701-234841
#
# Env knobs:
#   CLAUSE   : contract_execution_clause list body. Default 'cond, bpas' (what found it).
#   SSBD_OFF : 1 = ssbp patch OFF (default, leaks), 0 = ON (mitigated, should not reproduce).
#   TIMEOUT  : per-run wall-clock limit in seconds (default 900). 0 or 'none' = no timeout
#              (the model is bounded by model_max_spec_window/model_max_nesting, so it WILL end).
#   NI       : num inputs if the folder has none saved (default 100; ignored when input*.bin exist).
set -u

REVIZOR_DIR="/home/mluo/sca-fuzzer"
VIOL_DIR="$REVIZOR_DIR/rvzr/SpecRL/hierarchical_testing"
BASE_CFG="base.json"                       # ISA spec (-s), relative to REVIZOR_DIR
GEN_CFG="$VIOL_DIR/big_fuzz.yaml"           # the real generation config

name="${1:?usage: ./reproduce_one.sh <violation-folder-name> [num_inputs]}"
NI="${2:-${NI:-100}}"
CLAUSE="${CLAUSE:-cond, bpas}"
SSBD_OFF="${SSBD_OFF:-1}"
TIMEOUT="${TIMEOUT:-900}"
IPC="${IPC:-1}"          # inputs_per_class. 1 = NO boosting/taint-tracking (fast, and enough for
                         # reproduction: the violating inputs are already in the saved set). Only set
                         # >1 if you specifically want boosting (WARNING: enables per-rollback
                         # copy.deepcopy of the taint state -> can be extremely slow on store-heavy code).

F="$VIOL_DIR/$name"
asm="$F/program.asm"
[ -f "$asm" ]     || { echo "missing $asm"; exit 1; }
[ -f "$GEN_CFG" ] || { echo "missing $GEN_CFG"; exit 1; }
cd "$REVIZOR_DIR" || { echo "cannot cd $REVIZOR_DIR"; exit 1; }

# effective config = big_fuzz.yaml with the ssbp + contract-clause + inputs_per_class
# lines stripped, then re-appended to the values we want.
cfg="$(mktemp)"
grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^[[:space:]]*inputs_per_class:|^[[:space:]]*-[[:space:]]*(seq|bpas|cond|cond-bpas)[[:space:]]*$' "$GEN_CFG" > "$cfg"
# keep the original per-folder seed if present (harmless; -i overrides inputs anyway)
[ -f "$F/reproduce.yaml" ] && grep -E '^[a-zA-Z].*seed' "$F/reproduce.yaml" >> "$cfg"
printf 'contract_execution_clause: [%s]\n' "$CLAUSE" >> "$cfg"
printf 'inputs_per_class: %s\n' "$IPC" >> "$cfg"
if [ "$SSBD_OFF" = "1" ]; then
    echo 'x86_executor_enable_ssbp_patch: false' >> "$cfg"
else
    echo 'x86_executor_enable_ssbp_patch: true'  >> "$cfg"
fi

# inputs: prefer the saved ones, else generate NI from the seed
shopt -s nullglob
inputs=( "$F"/input_*.bin )
shopt -u nullglob
if [ ${#inputs[@]} -gt 0 ]; then
    in_args=( -i "${inputs[@]}" ); in_desc="${#inputs[@]} saved input*.bin"
else
    in_args=( -n "$NI" );          in_desc="$NI generated inputs (seed from cfg)"
fi

ssb_sys="$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)"
log="$F/reproduce_one.log"
echo "folder     : $name"
echo "contract   : ct + [$CLAUSE]   inputs_per_class=$IPC   ssbp_patch_off=$SSBD_OFF   SSBD(system)=$ssb_sys"
echo "inputs     : $in_desc"
echo "cmd        : ./revizor.py reproduce -s $BASE_CFG -c <cfg> -t program.asm ${in_args[0]} ..."
echo "log        : $log   (timeout ${TIMEOUT}s)"
echo "------------------------------------------------------------"

if [ "$TIMEOUT" = "0" ] || [ "$TIMEOUT" = "none" ]; then
    # no timeout: let the (bounded) model finish; Ctrl+C to abort manually
    ./revizor.py reproduce \
        -s "$BASE_CFG" -c "$cfg" -t "$asm" "${in_args[@]}" > "$log" 2>&1
    rc=$?
else
    timeout --signal=INT "$TIMEOUT" ./revizor.py reproduce \
        -s "$BASE_CFG" -c "$cfg" -t "$asm" "${in_args[@]}" > "$log" 2>&1
    rc=$?
fi
rm -f "$cfg"

nviol="$(grep -aoiE 'violations?: *[0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)"
if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then res="TIMEOUT (inconclusive)"
elif grep -aqiE 'no more checkpoints|Traceback|AssertionError|Exception' "$log"; then res="ERROR (slow-path crash; try CLAUSE=bpas)"
elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then res="NO  (does NOT reproduce)"
elif grep -aqiE 'violation' "$log"; then res="YES (REPRODUCED)"
else res="? (see log)"; fi

echo "RESULT: $res    [n_violations=${nviol:-?}]  rc=$rc"
[ "${res:0:3}" = "ERR" ] && { echo '--- last 15 log lines ---'; tail -15 "$log"; }
