#!/usr/bin/env bash
# Reproduce ONE violation N times and tally how often it is REAL (survives
# priming) vs a false positive. Mirrors:
#   ./revizor.py reproduce -s base.json -c big_fuzz.yaml -t <F>/program.asm -i <F>/input*.bin
# but forces SSBD OFF (big_fuzz sets it ON -> can never reproduce), which is the
# condition the violation was found under.
#
# Env knobs:
#   N=20            number of repetitions
#   SSBD_OFF=1      1=force SSBD off (default; required to reproduce). 0=big_fuzz as-is (SSBD on).
#   PER_TIMEOUT=3600  per-run timeout (s). Full pipeline can take ~40min/run.
#   FAST=0          1=only run to the fast path (skip slow-path priming) -> fast but
#                   counts raw fast-path hits (includes noise), NOT "survives priming".
# Usage: ./repeat_reproduce.sh <violation-folder>
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:?usage: ./repeat_reproduce.sh <violation-folder>}"; F="$VD/$name"
N="${N:-20}"; SSBD_OFF="${SSBD_OFF:-1}"; PER_TIMEOUT="${PER_TIMEOUT:-3600}"; FAST="${FAST:-0}"
cd "$RV" || exit 1
[ -f "$F/program.asm" ] || { echo "no program.asm"; exit 1; }
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
smt=$(cat /sys/devices/system/cpu/smt/active 2>/dev/null)
echo "folder=$name  N=$N  SSBD_OFF=$SSBD_OFF  FAST=$FAST  system spec_store_bypass=$ssb  SMT_active=$smt"
[ "$smt" = 1 ] && echo "WARNING: SMT is ON -> P+P false positives likely. Consider: echo off | sudo tee /sys/devices/system/cpu/smt/control"

# build config: big_fuzz + (optional) SSBD off + (optional) FAST tweaks
cfg="$F/repeat_cfg.yaml"
if [ "$SSBD_OFF" = 1 ]; then
  grep -v '^[[:space:]]*x86_executor_enable_ssbp_patch:' "$VD/big_fuzz.yaml" > "$cfg"
  echo 'x86_executor_enable_ssbp_patch: false' >> "$cfg"
else
  cp "$VD/big_fuzz.yaml" "$cfg"
fi
if [ "$FAST" = 1 ]; then
  # fast path only: disable the slow-path FP filters (raw fast-path hit)
  grep -vE '^[[:space:]]*(enable_speculation_filter|enable_observation_filter|enable_priming):' "$cfg" > "$cfg.tmp" && mv "$cfg.tmp" "$cfg"
  printf 'enable_priming: false\n' >> "$cfg"
fi

OUT="$F/repeat_reproduce.tsv"; printf 'run\tverdict\tviol\tdur_s\n' > "$OUT"
real=0; fp=0; noleak=0; tmo=0; err=0
for i in $(seq 1 "$N"); do
  log="$F/repeat_$i.log"
  timeout --signal=INT "$PER_TIMEOUT" ./revizor.py reproduce -s base.json -c "$cfg" \
      -t "$F/program.asm" -i "$F"/input*.bin > "$log" 2>&1
  rc=$?
  viol=$(grep -aoiE '^Violations: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
  dur=$(grep -aoE 'Duration: [0-9.]+' "$log" | grep -aoE '[0-9.]+' | tail -1)
  prim=$(grep -aoE 'Priming Check: [0-9]+' "$log" | grep -aoE '[0-9]+' | tail -1)
  if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then v=TIMEOUT; tmo=$((tmo+1))
  elif grep -aqiE 'no more checkpoints|Traceback|AssertionError' "$log"; then v=ERROR; err=$((err+1))
  elif [ "${viol:-0}" -ge 1 ] 2>/dev/null; then v=REAL; real=$((real+1))
  elif [ "${prim:-0}" -ge 1 ] 2>/dev/null; then v=FP-priming; fp=$((fp+1))
  else v=NO-LEAK; noleak=$((noleak+1)); fi
  printf '%s\t%s\t%s\t%s\n' "$i" "$v" "${viol:-?}" "${dur:-?}" >>"$OUT"
  echo "[$i/$N] -> $v (Violations=${viol:-?}, ${dur:-?}s)"
done
rm -f "$cfg"
echo "================================================================"
echo "REAL(survives priming, Violations>=1) = $real"
echo "FP-priming(discarded by priming)      = $fp"
echo "NO-LEAK(spec-filter/instant)          = $noleak"
echo "TIMEOUT(slow path unfinished)         = $tmo"
echo "ERROR                                 = $err   / N=$N"
echo "Detail: $OUT"
