#!/usr/bin/env bash
#
# replay_fuzz.sh — replay a saved multi-actor FUZZING violation via `revizor.py reproduce`, N times.
#
# Defaults to the guest-user -> guest-user fault-driven (L1TF/Foreshadow-style) cross-actor leak
# found at TC 19779. Uses the violation's own saved config (reproduce.yaml: ct-ni, F+R, hpa_gpa on),
# program, and 100 inputs. exit 1 = violation reproduced, 0 = clean.
#
# Needs the CLEAN executor build (1CD139EF) + perf_env.sh (F+R is noise-sensitive). NOT the IBPB build
# (that only scrubs branch predictors — irrelevant to this fault-based leak, but use clean to match).
#
# Usage:
#   bash replay_fuzz.sh                         # N=5 on the u2u violation
#   N=10 bash replay_fuzz.sh
#   V=<other-violation-dir> bash replay_fuzz.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-fuzz-u2u/violation-260814-125938}
CONFIG=${CONFIG:-$V/reproduce.yaml}   # override e.g. CONFIG=$CAMP/config-fuzz-u2k-strong.yaml
N=${N:-5}

[ -f "$V/program.asm" ] || { echo "no program.asm in $V"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
SV=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)

# --- hpa_gpa EPT-collision pre-check ---
# With x86_enable_hpa_gpa_collisions on, a colliding module layout makes EVERY test-case load
# fail (set_extended_page_tables assertion -> EIO / Duration 0.0 / always-clean). Detect it up front
# so a "0/N" here is a real weak signal, not a broken executor.
b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
timeout 90 "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$V/program.asm" -i "$V"/input_*.bin >/dev/null 2>&1 || true
a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
if [ "$a" -gt "$b" ]; then
    echo "!! hpa_gpa EPT COLLISION — this module load can't map the sandbox (every load EIOs)."
    echo "   Fix: bash $CAMP/reload_until_ok.sh   (reloads until a non-colliding layout), then re-run."
    exit 3
fi

echo "=================================================================="
echo " REPLAY fuzzing violation x$N"
echo "   violation : $V"
echo "   config    : $CONFIG"
echo "   module    : $SV   (want clean 1CD139EF)"
echo "   inputs    : $(ls "$V"/input_*.bin | wc -l) saved inputs"
echo "=================================================================="

hits=0
for i in $(seq 1 "$N"); do
    "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$V/program.asm" -i "$V"/input_*.bin \
        >/dev/null 2>&1
    rc=$?
    if [ "$rc" -eq 1 ]; then hits=$((hits+1)); echo "  run $i: exit=1  VIOLATION reproduced";
    elif [ "$rc" -eq 0 ]; then echo "  run $i: exit=0  clean";
    else echo "  run $i: exit=$rc ERROR"; fi
done

echo "=================================================================="
echo " RESULT: $hits/$N reproduced   (module $SV)"
[ "$hits" -gt 0 ] && echo " => the cross-actor fault leak reproduces." \
                  || echo " => did not reproduce this round (weak/marginal signal — raise N, check perf_env.sh)."
echo "=================================================================="
