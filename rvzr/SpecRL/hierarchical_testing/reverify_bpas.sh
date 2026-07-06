#!/usr/bin/env bash
# Re-judge the 89 cond-bpas "violations" under PLAIN bpas (ONLY the contract
# changes; everything else identical to the run that produced them:
# program.asm + original input*.bin + ct obs + SSBD off + same filters).
#   NO  = plain bpas explains it -> it was a cond-bpas-bug false positive (plain v4)
#   YES = survives plain bpas    -> genuinely beyond bpas (real candidate)
# Reads the YES list from verify_real_condbpas.tsv.
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
TIMEOUT="${TIMEOUT:-240}"; OUT="$VD/reverify_bpas.tsv"
cd "$RV" || exit 1
mapfile -t YES < <(awk -F'\t' '$2=="YES"{print $1}' "$VD/verify_real_condbpas.tsv")
tot=${#YES[@]}
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
[ "$ssb" = Vulnerable ] || echo "WARNING: system SSBD not off ($ssb)"
printf 'folder\tbpas\tn\n' > "$OUT"
echo "Re-judging $tot folders under PLAIN bpas (SSBD off). -> $OUT"
no=0; yes=0; err=0; tmo=0; i=0
for name in "${YES[@]}"; do
  d="$VD/$name"; i=$((i+1))
  [ -f "$d/program.asm" ] || { printf '%s\tSKIP\t-\n' "$name">>"$OUT"; echo "[$i/$tot] $name SKIP"; continue; }
  # same config as verify_real (big_fuzz minus ssbp/contract, + seed) but CLAUSE=bpas
  cfg=$(mktemp)
  grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$cfg"
  grep -E '^[a-zA-Z].*seed' "$d/reproduce.yaml" >> "$cfg"
  printf 'x86_executor_enable_ssbp_patch: false\ncontract_execution_clause: [bpas]\n' >> "$cfg"
  log="$d/reverify_bpas.log"
  timeout --signal=INT "$TIMEOUT" ./revizor.py reproduce -s base.json -c "$cfg" \
      -t "$d/program.asm" -i "$d"/input*.bin > "$log" 2>&1
  rc=$?; rm -f "$cfg"
  nv=$(grep -aoiE 'violations?: *([0-9]+)' "$log" | grep -aoE '[0-9]+' | tail -1)
  if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then v=TIMEOUT; tmo=$((tmo+1))
  elif grep -aqiE 'no more checkpoints|Traceback|AssertionError' "$log"; then v=ERROR; err=$((err+1))
  elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then v=NO; no=$((no+1))
  elif grep -aqiE 'violation' "$log"; then v=YES; yes=$((yes+1))
  else v="?"; fi
  printf '%s\t%s\t%s\n' "$name" "$v" "${nv:-?}" >>"$OUT"
  echo "[$i/$tot] $name -> bpas=$v"
done
echo "================================================================"
echo "Under PLAIN bpas:  YES(survives=real beyond-bpas)=$yes   NO(false-positive=plain v4)=$no   ERROR=$err   TIMEOUT=$tmo   / $tot"
echo "Detail: $OUT"
