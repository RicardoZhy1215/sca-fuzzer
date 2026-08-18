#!/usr/bin/env bash
#
# replay_all.sh — reproduce EVERY violation in a directory, N times each, and print a summary table.
#
# For each violation-*/ under DIR, runs `revizor.py reproduce` N times using that violation's OWN
# saved config (reproduce.yaml), program.asm, and saved inputs. exit 1 = reproduced, 0 = clean.
# The per-violation rate tells you which finds are solid vs drift.
#
# Needs the CLEAN executor build (1CD139EF) + perf_env.sh (F+R is noise-sensitive), and NO other
# revizor process touching the executor at the same time (concurrent access wedges it).
#
# Usage:
#   bash replay_all.sh                                   # all of out-refuzz-u2k, N=10 each
#   DIR=campaigns/vmscape-repro/out-refuzz-u2u bash replay_all.sh
#   N=20 DIR=campaigns/vmscape-repro/out-fuzz-u2k bash replay_all.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
DIR=${DIR:-$CAMP/out-refuzz-u2k}
N=${N:-10}

[ -d "$DIR" ] || { echo "no such directory: $DIR"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
SV=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)
cd "$REPO"

shopt -s nullglob
VIOLS=( "$DIR"/violation-* )
[ "${#VIOLS[@]}" -gt 0 ] || { echo "no violation-* dirs in $DIR"; exit 2; }

echo "=================================================================="
echo " REPLAY ALL violations in $DIR"
echo "   count  : ${#VIOLS[@]} violations   x N=$N each"
echo "   module : $SV   (want clean 1CD139EF)"
echo "=================================================================="

declare -a SUMMARY=()
for V in $(ls -dt "$DIR"/violation-* 2>/dev/null); do
    CONFIG=$V/reproduce.yaml
    name=$(basename "$V")
    if [ ! -f "$V/program.asm" ]; then
        echo "--- $name : SKIP (no program.asm) ---"; SUMMARY+=("$name: skip"); continue
    fi
    [ -f "$CONFIG" ] || CONFIG=$CAMP/config-refuzz-u2k.yaml   # fallback
    thr=$(grep -E 'analyser_stat_threshold' "$CONFIG" 2>/dev/null | awk '{print $2}')
    nin=$(ls "$V"/input_*.bin 2>/dev/null | wc -l)
    echo "--- $name : x$N   (thr=$thr, inputs=$nin) ---"
    hits=0
    for i in $(seq 1 "$N"); do
        b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$V/program.asm" -i "$V"/input_*.bin >/dev/null 2>&1
        rc=$?
        a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        if [ "$a" -gt "$b" ]; then
            echo "  !! hpa_gpa EPT COLLISION — this layout can't map the sandbox (every load EIOs)."
            echo "     Fix: bash $CAMP/reload_until_ok.sh   then re-run."; exit 3
        fi
        [ "$rc" -eq 1 ] && hits=$((hits+1))
        printf "  run %2d: exit=%d  %s\n" "$i" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    done
    echo "  => $name reproduced $hits/$N"
    SUMMARY+=("$name: $hits/$N")
done

echo "=================================================================="
echo " SUMMARY — reproduction rate per violation ($DIR):"
printf '   %s\n' "${SUMMARY[@]}"
echo "=================================================================="
