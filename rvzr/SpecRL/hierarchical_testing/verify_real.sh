#!/usr/bin/env bash
# Verify whether violations are REAL (not flaky): replay each under an EXPLICIT
# contract (NOT the mutable big_fuzz contract), SSBD OFF, full FP pipeline
# (nesting / taint / priming / large-sample). Survives -> real contract
# violation; vanishes -> flaky/noise (or explained by the contract).
#
# Usage:
#   CLAUSE=seq        ./verify_real.sh 'violation-260619-*' 'violation-260620-*'   # realness test
#   CLAUSE='cond, bpas' ./verify_real.sh 'violation-260619-*' 'violation-260620-*' # classify vs cond-bpas
#   CLAUSE=bpas       ./verify_real.sh 'violation-260619-*'                         # classify vs bpas
#
# Interpretation:
#   under seq:   YES = REAL (genuine speculative leak, survives priming);  NO = flaky/noise
#   under bpas / cond-bpas: YES = beyond that contract; NO = explained by it (known V4 / V1+V4)
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
CLAUSE="${CLAUSE:-seq}"            # explicit exec clause; "seq" | "bpas" | "cond, bpas"
TIMEOUT="${TIMEOUT:-240}"          # per-folder timeout (s)
[ $# -ge 1 ] || { echo "usage: CLAUSE=seq $0 '<glob>' ['<glob>' ...]"; exit 1; }
tag=$(echo "$CLAUSE" | tr -cd 'a-z0-9')
OUT="$VD/verify_real_${tag}.tsv"
cd "$RV" || exit 1

# collect folders from ALL given globs (dedup, sorted)
shopt -s nullglob
declare -A seen; folders=()
for pat in "$@"; do
  for d in "$VD"/$pat/; do
    d=${d%/}; [ -n "${seen[$d]:-}" ] && continue; seen[$d]=1; folders+=("$d")
  done
done
shopt -u nullglob
tot=${#folders[@]}
[ "$tot" -gt 0 ] || { echo "no folders matched: $*"; exit 1; }

# sanity: SSBD must be off (Vulnerable) to match training, else everything is trivially NO
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
echo "SSBD(system spec_store_bypass)=$ssb | CLAUSE=[$CLAUSE] | folders=$tot | -> $OUT"
[ "$ssb" = "Vulnerable" ] || echo "WARNING: SSBD not 'Vulnerable'; store-bypass leaks may not appear (results may be trivially NO)."

printf 'folder\tverdict\tn\n' > "$OUT"
yes=0; no=0; err=0; tmo=0; i=0
for d in "${folders[@]}"; do
  n=$(basename "$d"); i=$((i+1))
  [ -f "$d/program.asm" ] || { printf '%s\tSKIP\t-\n' "$n">>"$OUT"; echo "[$i/$tot] $n SKIP"; continue; }
  cfg=$(mktemp)
  grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$cfg"
  grep -E '^[a-zA-Z].*seed' "$d/reproduce.yaml" >> "$cfg"
  echo "x86_executor_enable_ssbp_patch: false" >> "$cfg"
  echo "contract_execution_clause: [$CLAUSE]" >> "$cfg"
  log="$d/verify_${tag}.log"
  timeout --signal=INT "$TIMEOUT" ./revizor.py reproduce -s base.json -c "$cfg" \
      -t "$d/program.asm" -i "$d"/input*.bin > "$log" 2>&1
  rc=$?; rm -f "$cfg"
  nv=$(grep -aoiE 'violations?: *([0-9]+)' "$log" | grep -aoE '[0-9]+' | tail -1)
  if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then v=TIMEOUT; tmo=$((tmo+1))
  elif grep -aqiE 'no more checkpoints|Traceback|AssertionError' "$log"; then v=ERROR; err=$((err+1))
  elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then v=NO; no=$((no+1))
  elif grep -aqiE 'violation' "$log"; then v=YES; yes=$((yes+1))
  else v="?"; fi
  printf '%s\t%s\t%s\n' "$n" "$v" "${nv:-?}" >>"$OUT"
  echo "[$i/$tot] $n -> $v"
done
echo "==== CLAUSE=[$CLAUSE] SSBD=off : YES=$yes NO=$no ERROR=$err TIMEOUT=$tmo / $tot ===="
echo "detail: $OUT"
