#!/usr/bin/env bash
#
# repro_tsa_sq.sh — reproduce TSA-SQ (AMD Transient Scheduler Attack, Store-Queue variant).
#
#   TSA-SQ = store-to-load-forwarding FALSE COMPLETION: a user load matches an older KERNEL store's
#   address, but the store data isn't yet available, so the load false-completes with STALE data from
#   a previous store that occupied the same store-queue entry -> leaks a recently-stored kernel value
#   to user mode. Boundary = K2U (kernel -> user), both HOST actors, syscall/sysret transitions.
#
#   This drives the repo's canonical demo assets UNMODIFIED (demo/tsa-sq/{config,template}):
#     - main  = host KERNEL  (victim): "secret injection" (fuzzed stores)
#     - user  = host USER    (observer): "secret retrieval" (loads that hit the store-queue) + F+R
#   The demo uses FIXED seeds (program 20000000 / data 1000000) so tfuzz deterministically walks into
#   the TSA-SQ gadget, and threshold 0.2 (lower than tsa-l1d's 0.5 — the SQ signal is smaller).
#
#   Reproduces on TSA-VULNERABLE microcode. This box is currently 0xa705206 (reverted/pre-fix) -> good.
#   (On 0208 the AMD microtag/SQ mitigation is VERW-gated; the executor's uarch_flush is a stub so it
#    may still fire — but 0206 is the clean repro.)
#
# Usage:
#   bash repro_tsa_sq.sh                 # demo defaults: N=10000 x I=50, stop at first violation
#   N=50000 bash repro_tsa_sq.sh         # longer walk if the default seeds miss on your box
#   NONSTOP=1 bash repro_tsa_sq.sh       # keep going after the first hit (collect several)
#   THR=0.1 bash repro_tsa_sq.sh         # loosen the stat threshold if the SQ signal is weak/noisy
#   SMOKE=1 bash repro_tsa_sq.sh         # 10-TC pipeline check
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
TEMPLATE=$REPO/demo/tsa-sq/template.asm
SRC_CONFIG=$REPO/demo/tsa-sq/config.yaml
N=${N:-10000}; I=${I:-50}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && { N=10; I=10; }
OUT=$CAMP/out-tsa-sq
NONSTOP=${NONSTOP:-0}; NS_FLAG=(); [ "$NONSTOP" = "1" ] && NS_FLAG=(--nonstop)

# use the demo config as-is, unless THR overrides the threshold (then work on a copy — keep demo clean)
CONFIG=$SRC_CONFIG
if [ -n "${THR:-}" ]; then
    CONFIG=$CAMP/config-tsa-sq.yaml
    sed "s/^analyser_stat_threshold:.*/analyser_stat_threshold: $THR/" "$SRC_CONFIG" > "$CONFIG"
fi

echo "=================================================================="
echo " REPRODUCE TSA-SQ  (store-queue false-completion, K2U: kernel -> user)"
echo "   template : $TEMPLATE"
echo "   config   : $CONFIG   (thr $(grep -E '^analyser_stat_threshold' "$CONFIG" | awk '{print $2}'), K2U host-host, fixed seeds)"
echo "   campaign : N=$N x I=$I   nonstop=$NONSTOP   (SMOKE=$SMOKE)"
echo "=================================================================="
[ -f "$TEMPLATE" ] || { echo "no template at $TEMPLATE"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "[mod] module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
echo "[mod] loaded: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
echo "[ucode] $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)   (TSA-SQ needs a vulnerable ucode; 0xa705206 = pre-fix)"
echo "[env] governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)  (want performance/off; run perf_env.sh)"

mkdir -p "$OUT"
echo "[run] tfuzz starting ($(date +%H:%M:%S)) — live log:"
echo "------------------------------------------------------------------"
cd "$REPO"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT" "${NS_FLAG[@]}"
RC=$?
echo "------------------------------------------------------------------"
shopt -s nullglob
Vs=( "$OUT"/violation-* )
echo " violations: ${#Vs[@]}   (in $OUT)"
if [ "${#Vs[@]}" -gt 0 ]; then
    LATEST=$(ls -dt "$OUT"/violation-* | head -1)
    echo " => TSA-SQ reproduced. Latest: $LATEST"
    echo "    next: minimize it, then leaked-values. The leaked byte should decode to the"
    echo "    KERNEL actor (main, actor 0); the observer(user)'s input must be byte-identical"
    echo "    across the counterexample (pure store-queue value-forward, no arch access differs)."
else
    echo " => no violation at N=$N. If the box is TSA-vulnerable (0206) this is likely a threshold/"
    echo "    noise issue: run perf_env.sh, then THR=0.1 and/or raise N. If on 0208 with the SQ"
    echo "    mitigation active, it may be genuinely closed."
fi
echo "=================================================================="
exit "$RC"
