#!/usr/bin/env bash
# Root-cause a minimized violation; confirm partial store-to-load forwarding.
# Fast clincher first (de-align), then the slower fence / input-diff passes.
# Usage:  ./root_cause.sh <violation-folder> [num_inputs]
set -u
RV=/home/mluo/sca-fuzzer; VD=$RV/rvzr/SpecRL/hierarchical_testing
name="${1:?usage: ./root_cause.sh <violation-folder> [num_inputs]}"
F="$VD/$name"; NI="${2:-100}"
MIN_TO="${MIN_TO:-2400}"   # timeout (s) for the slow minimize passes
cd "$RV" || exit 1
cfg="$F/minimize_test.yaml"; base="$F/minimized.asm"
[ -f "$base" ] && [ -f "$cfg" ] || { echo "need minimized.asm + minimize_test.yaml"; exit 1; }
export RVZR_MINIMIZE_INPUT_DIR="$F"
ssb=$(cat /sys/devices/system/cpu/vulnerabilities/spec_store_bypass 2>/dev/null)
echo "folder=$name  SSBD(system)=$ssb"; grep -E 'contract_|cond|bpas|seq|ssbp' "$cfg"

# reproduce <asm> <tag> -> YES/NO/ERROR/TIMEOUT   (uses min_inputs if present, else original)
repro() {
  local asm="$1"
  local tag="$2"
  local log="$F/rc_${tag}.log"
  local ins
  if ls "$F"/min_inputs/input*.bin >/dev/null 2>&1; then ins=( "$F"/min_inputs/input*.bin ); else ins=( "$F"/input*.bin ); fi
  timeout --signal=INT 300 ./revizor.py reproduce -s base.json -c "$cfg" -t "$asm" -i "${ins[@]}" > "$log" 2>&1
  local rc=$?
  if   [ "$rc" -eq 124 ] || grep -aqiE 'KeyboardInterrupt' "$log"; then echo TIMEOUT
  elif grep -aqiE 'no more checkpoints|Traceback|AssertionError' "$log"; then echo ERROR
  elif grep -aqiE 'no violation|violations: 0|0 violations|not reproduc' "$log"; then echo NO
  elif grep -aqiE 'violation' "$log"; then echo YES
  else echo "?"; fi
}

echo; echo "################ STEP A (fast): de-align clincher ################"
B=$(repro "$base" baseline)
sed 's/0b1111111111111/0b1111111111000/g' "$base" > "$F/aligned.asm"
A=$(repro "$F/aligned.asm" aligned)
echo "  baseline (unaligned/line-split allowed) : $B"
echo "  de-aligned (full masks -> 8B aligned)   : $A"
if [ "$B" = YES ] && [ "$A" = NO ]; then echo "  >>> de-align KILLS it: partial/line-split overlap is CAUSAL => partial STLF"
elif [ "$B" = YES ] && [ "$A" = YES ]; then echo "  >>> survives de-align: NOT alignment-driven; see fence/input-diff below"
else echo "  >>> baseline=$B aligned=$A (see rc_baseline.log / rc_aligned.log)"; fi

echo; echo "################ STEP B: fence pass (speculation window) ################"
timeout "$MIN_TO" ./revizor.py minimize -s base.json -c "$cfg" -t "$base" -o "$F/fenced.asm" -i "$NI" \
  --enable-instruction-pass false --enable-label-pass false --enable-fence-pass true \
  > "$F/rc_fence.log" 2>&1
if [ -f "$F/fenced.asm" ]; then
  echo "  LFENCEs inserted (violation preserved): $(grep -aciE '\blfence\b' "$F/fenced.asm")  -> fenced.asm"
  echo "  gaps with NO lfence after a mem op = speculation window. Real instrs:"
  grep -avE '^\s*(#|\.)|instrumentation' "$F/fenced.asm" | grep -aE 'lfence|r14|cmov|mov|add|cmpxchg' | head -40
else
  echo "  fence pass did not finish within ${MIN_TO}s (see rc_fence.log tail):"; tail -4 "$F/rc_fence.log"
fi

echo; echo "################ STEP C: input-diff (leaked bytes ^ / enabling = +) ################"
timeout "$MIN_TO" ./revizor.py minimize -s base.json -c "$cfg" -t "$base" -o "$F/diff.asm" -i "$NI" \
  --enable-instruction-pass false --enable-label-pass false \
  --enable-input-diff-pass true --input-outdir "$F/min_inputs" \
  > "$F/rc_diff.log" 2>&1
echo "  min_inputs: $(ls "$F"/min_inputs/input*.bin 2>/dev/null | wc -l)   (log: rc_diff.log)"
echo "  --- differential grid rows that carry markers (^ leaked, = + enabling) ---"
grep -aE '0x[0-9a-f]{8}' "$F/rc_diff.log" | grep -aE '[\^=+]' | head -40 || echo "  (no marker rows; tail of log:)"
grep -aE '[\^=+]|leak|Minimizing the difference' "$F/rc_diff.log" | tail -20

echo; echo "################ VERDICT ################"
echo "  de-align: baseline=$B aligned=$A | SSBD(system)=$ssb | contract=cond-bpas (see ssbp above)"
echo "  Files: aligned.asm fenced.asm diff.asm + rc_*.log in $F"
