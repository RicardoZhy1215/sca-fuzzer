#!/usr/bin/env bash
# PSFD discriminator: is violation-260701-234841 (minimized) PSF or classic SSB?
#
# REQUIRES the executor rebuilt with the new enable_psfd knob (SPEC_CTRL bit 7):
#   cd /home/mluo/sca-fuzzer/rvzr/executor_km && make uninstall; make && make install
# After reload, /sys/rvzr_executor/enable_psfd exists (world-writable, default 0).
#
# Runs the minimized gadget under 3 MSR conditions (full priming), N times each:
#   A baseline  SSBD=0 PSFD=0  -> expect violation PRESENT
#   B ssbd      SSBD=1 PSFD=0  -> expect GONE (SSBD mitigates SSB, and usually PSF too)
#   C psfd      SSBD=0 PSFD=1  -> DECISIVE (SSB still ON, only PSF disabled):
#        GONE    => PSF  (Predictive Store Forwarding) — PSFD-specific mitigation hits it
#        PRESENT => NOT PSF; a classic SSB-family store-bypass variant bpas doesn't model
#
# SSBD is set per-run via the yaml key x86_executor_enable_ssbp_patch (revizor writes it).
# PSFD is set via sysfs echo here (revizor never touches enable_psfd, so it persists).
#
# Env: N=5  CLAUSE='cond, bpas'
# Usage: ./psfd_test.sh [violation-folder=violation-260701-234841]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:-violation-260701-234841}"; F="$VD/$name"
N="${N:-5}"; CLAUSE="${CLAUSE:-cond, bpas}"
PSFD_SYS=/sys/rvzr_executor/enable_psfd
cd "$RV" || exit 1

# --- preflight -------------------------------------------------------------
asm="$F/minimized.asm"; [ -f "$asm" ] || asm="$F/program.asm"
[ -f "$asm" ] || { echo "ERROR: no minimized.asm/program.asm in $F"; exit 1; }
if [ ! -e "$PSFD_SYS" ]; then
  echo "ERROR: $PSFD_SYS missing -> executor not rebuilt with the PSFD knob."
  echo "  Rebuild + reload (needs root):"
  echo "    cd $RV/rvzr/executor_km && make uninstall; make && make install"
  exit 1
fi
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
smt=$(cat /sys/devices/system/cpu/smt/active 2>/dev/null)
[ "$ssb" = Vulnerable ] || echo "NOTE: system spec_store_bypass=$ssb (per-run yaml still controls executor SSBD)"
[ "$smt" = 1 ] && echo "WARNING: SMT on -> extra P+P noise; consider turning it off."
if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then ins=( "$F"/min_inputs/input*.bin ); isrc=min_inputs
else ins=( "$F"/input*.bin ); isrc=input; fi
echo "asm=$(basename "$asm")  inputs=${#ins[@]} ($isrc)  N=$N  CLAUSE=[$CLAUSE]"

# --- config builder: $1 = ssbp value (true/false) -------------------------
mkcfg() {
  local ssbp="$1" out; out="$(mktemp)"
  grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^[[:space:]]*inputs_per_class:|^[[:space:]]*enable_priming:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$out"
  grep -E '^(data_generator_seed|inputs_per_class):' "$F/reproduce.yaml" >> "$out"
  { echo "x86_executor_enable_ssbp_patch: $ssbp"
    echo "contract_observation_clause: ct"
    echo "contract_execution_clause: [$CLAUSE]"
    echo "enable_priming: true"; } >> "$out"
  echo "$out"
}

# --- run one condition N times: $1=label $2=ssbp $3=psfd -------------------
run_cond() {
  local label="$1" ssbp="$2" psfd="$3" cfg real=0 tot=0 i log viol dur
  echo "$psfd" > "$PSFD_SYS" 2>/dev/null || { echo "  ERROR: cannot write $PSFD_SYS"; return 1; }
  cfg="$(mkcfg "$ssbp")"
  echo; echo "### $label  (SSBD=$([ "$ssbp" = true ] && echo 1 || echo 0)  PSFD=$psfd)  enable_psfd=$(cat "$PSFD_SYS") ###"
  for i in $(seq 1 "$N"); do
    log="$F/psfd_${label}_$i.log"
    ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
    viol=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
    dur=$(grep -aoE 'Duration: [0-9.]+' "$log" | grep -aoE '[0-9.]+' | tail -1)
    if [ -n "$viol" ] && [ "$viol" -ge 1 ] 2>/dev/null; then real=$((real+1)); fi
    tot=$((tot+1))
    echo "  [$label $i/$N] Violations=${viol:-?} (${dur:-?}s)"
  done
  rm -f "$cfg"
  echo "  => $label: violation PRESENT in $real / $tot runs"
  eval "RES_${label}=$real"
}

run_cond A_baseline false 0
run_cond B_ssbd     true  0
run_cond C_psfd     false 1
echo 0 > "$PSFD_SYS" 2>/dev/null   # leave PSFD off

# --- summary + verdict -----------------------------------------------------
a=${RES_A_baseline:-0}; b=${RES_B_ssbd:-0}; c=${RES_C_psfd:-0}
echo; echo "================= SUMMARY (present / $N) ================="
printf '  A baseline  SSBD=0 PSFD=0 : %s\n' "$a"
printf '  B ssbd      SSBD=1 PSFD=0 : %s\n' "$b"
printf '  C psfd      SSBD=0 PSFD=1 : %s\n' "$c"
echo "----------------------------------------------------------"
if [ "$a" -ge 1 ] && [ "$c" -eq 0 ]; then
  echo "VERDICT: PSFD kills it while SSB stays ON  =>  PSF (Predictive Store Forwarding)."
  echo "  (sanity-check the MSR write actually took effect:  dmesg | grep -i 'set MSR')"
elif [ "$a" -ge 1 ] && [ "$c" -ge 1 ]; then
  echo "VERDICT: survives PSFD  =>  NOT PSF; a classic SSB-family store-bypass variant"
  echo "         (bpas-unmodeled, e.g. partial/size-mismatch STLF-bypass)."
else
  echo "VERDICT: INCONCLUSIVE (baseline didn't reproduce) -> rerun with larger N / check flakiness."
fi
echo "logs: $F/psfd_*_*.log"
