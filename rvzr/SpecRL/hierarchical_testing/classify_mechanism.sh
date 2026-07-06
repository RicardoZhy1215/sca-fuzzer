#!/usr/bin/env bash
# Mechanism-classify the confirmed-real violations by two discriminating tests
# (per folder, program.asm + saved inputs, contract = ct+cond-bpas):
#   ssbd_on : reproduce with SSBD ON  -> NO = SSB/v4-family (SSBD mitigates)
#                                        YES = SSBD-immune (MDS-like / other)
#   dealign : reproduce with all unaligned qword masks forced 8B-aligned
#             (0b...111 -> 0b...000), SSBD off -> NO = partial/split-dependent
# Buckets:
#   PARTIAL-SSB   ssbd_on=NO,  dealign=NO   (like 040753: SSBD-mitigated + split-dependent)
#   ALIGNED-SSB   ssbd_on=NO,  dealign=YES  (SSBD-mitigated, alignment-independent)
#   SSBD-IMMUNE   ssbd_on=YES               (survives SSBD -> not v4/SSB)
#   FLAKY/ODD     baseline!=YES or errors
# Reads the YES list from verify_real_condbpas.tsv.  ~3 reproduces/folder.
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
TIMEOUT="${TIMEOUT:-240}"; OUT="$VD/classify_mechanism.tsv"
cd "$RV" || exit 1
mapfile -t YES < <(awk -F'\t' '$2=="YES"{print $1}' "$VD/verify_real_condbpas.tsv")
tot=${#YES[@]}
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
[ "$ssb" = Vulnerable ] || { echo "WARNING: system SSBD not off ($ssb) — results unreliable"; }
printf 'folder\tbaseline\tssbd_on\tdealign\tbucket\n' > "$OUT"
echo "Classifying $tot folders. SSBD(system)=$ssb -> $OUT"

mkcfg() { # $1=ssbp(true/false) -> prints temp cfg path
  local d="$1" ssbp="$2" cfg; cfg=$(mktemp)
  grep -vE '^[[:space:]]*x86_executor_enable_ssbp_patch:|^contract_execution_clause:|^[[:space:]]*inputs_per_class:|^  - (seq|bpas|cond|cond-bpas)$' "$VD/big_fuzz.yaml" > "$cfg"
  grep -E '^(data_generator_seed|inputs_per_class):' "$d/reproduce.yaml" >> "$cfg"
  printf 'x86_executor_enable_ssbp_patch: %s\ncontract_execution_clause: [cond, bpas]\n' "$ssbp" >> "$cfg"
  echo "$cfg"
}
repro() { # $1=cfg $2=asm $3=dir $4=tag -> YES/NO/ERROR/TIMEOUT
  local log="$3/cm_$4.log"
  timeout --signal=INT "$TIMEOUT" ./revizor.py reproduce -s base.json -c "$1" -t "$2" -i "$3"/input*.bin > "$log" 2>&1
  local rc=$?
  if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then echo TIMEOUT
  elif grep -aqiE 'no more checkpoints|Traceback|AssertionError' "$log"; then echo ERROR
  elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then echo NO
  elif grep -aqiE 'violation' "$log"; then echo YES; else echo "?"; fi
}

i=0
for d in "${YES[@]}"; do
  d="$VD/$d"; i=$((i+1)); name=$(basename "$d")
  [ -f "$d/program.asm" ] || { printf '%s\tSKIP\t-\t-\tSKIP\n' "$name">>"$OUT"; continue; }
  c0=$(mkcfg "$d" false); c1=$(mkcfg "$d" true)
  b=$(repro "$c0" "$d/program.asm" "$d" baseline)
  on=$(repro "$c1" "$d/program.asm" "$d" ssbdon)
  sed 's/0b1111111111111/0b1111111111000/g' "$d/program.asm" > "$d/aligned_full.asm"
  da=$(repro "$c0" "$d/aligned_full.asm" "$d" dealign)
  rm -f "$c0" "$c1"
  if   [ "$b" != YES ]; then bk=FLAKY/ODD
  elif [ "$on" = YES ]; then bk=SSBD-IMMUNE
  elif [ "$on" = NO ] && [ "$da" = NO ]; then bk=PARTIAL-SSB
  elif [ "$on" = NO ] && [ "$da" = YES ]; then bk=ALIGNED-SSB
  else bk="on=$on,da=$da"; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$b" "$on" "$da" "$bk" >>"$OUT"
  echo "[$i/$tot] $name  base=$b ssbd_on=$on dealign=$da -> $bk"
done
echo "==== mechanism buckets ===="
tail -n +2 "$OUT" | cut -f5 | sort | uniq -c | sort -rn
