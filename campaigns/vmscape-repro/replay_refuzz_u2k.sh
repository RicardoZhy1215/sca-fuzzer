#!/usr/bin/env bash
#
# replay_refuzz_u2k.sh — reproduce a violation found by the refuzz-u2k campaign, N times.
#
# Defaults to the LATEST violation in out-refuzz-u2k/ and uses that violation's OWN saved config
# (reproduce.yaml: ct-ni, F+R, hpa_gpa on, analyser_stat_threshold 0.5), program, and 100 inputs.
# exit 1 = violation reproduced, 0 = clean. Reports the rate over N runs.
#
# Needs the CLEAN executor build (1CD139EF) + perf_env.sh (F+R is noise-sensitive), and NO other
# revizor process touching the executor at the same time.
#
# Usage:
#   bash replay_refuzz_u2k.sh                    # N=10 on the latest refuzz-u2k violation
#   N=20 bash replay_refuzz_u2k.sh
#   V=campaigns/.../out-refuzz-u2k/violation-260817-090802 bash replay_refuzz_u2k.sh   # a specific one
#   ALL=1 bash replay_refuzz_u2k.sh              # reproduce EVERY refuzz-u2k violation N times each
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
OUTDIR=$CAMP/out-refuzz-u2k
N=${N:-10}
ALL=${ALL:-0}

[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }
SV=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)
cd "$REPO"

replay_one () {   # $1 = violation dir ; echoes "hits/N", sets HITS
    local V=$1 CONFIG hits=0 rc b a
    CONFIG=${CONFIG_OVERRIDE:-$V/reproduce.yaml}
    [ -f "$V/program.asm" ] || { echo "  no program.asm in $V"; HITS=-1; return; }

    echo "=================================================================="
    echo " REPLAY  $V   x$N"
    echo "   config : $CONFIG   ($(grep -E 'analyser_stat_threshold' "$CONFIG" 2>/dev/null || echo 'thr n/a'))"
    echo "   module : $SV   (want clean 1CD139EF)   inputs: $(ls "$V"/input_*.bin 2>/dev/null | wc -l)"
    echo "------------------------------------------------------------------"
    for i in $(seq 1 "$N"); do
        # per-run hpa_gpa EPT-collision guard (a colliding layout EIOs every load -> false 0/N)
        b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$V/program.asm" -i "$V"/input_*.bin >/dev/null 2>&1
        rc=$?
        a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        if [ "$a" -gt "$b" ]; then
            echo "  !! hpa_gpa EPT COLLISION — this layout can't map the sandbox (every load EIOs)."
            echo "     Fix: bash $CAMP/reload_until_ok.sh   then re-run."; exit 3
        fi
        [ "$rc" -eq 1 ] && hits=$((hits+1))
        printf "  run %2d: exit=%d  %s\n" "$i" "$rc" "$([ "$rc" -eq 1 ] && echo 'VIOLATION reproduced' || { [ "$rc" -eq 0 ] && echo clean || echo ERROR; })"
    done
    HITS=$hits
    echo "  => reproduced $hits/$N"
}

if [ "$ALL" = "1" ]; then
    echo "### ALL refuzz-u2k violations, x$N each ###"
    declare -a SUMMARY=()
    for V in $(ls -dt "$OUTDIR"/violation-* 2>/dev/null); do
        replay_one "$V"; SUMMARY+=("$(basename "$V"): $HITS/$N")
    done
    echo "=================================================================="
    echo " SUMMARY (reproduction rate per violation):"
    printf '   %s\n' "${SUMMARY[@]}"
    echo "=================================================================="
    exit 0
fi

V=${V:-$(ls -dt "$OUTDIR"/violation-* 2>/dev/null | head -1)}
[ -n "$V" ] || { echo "no violations in $OUTDIR — run refuzz_u2k.sh first"; exit 2; }
replay_one "$V"
echo "=================================================================="
echo " RESULT: $HITS/$N reproduced   ($(basename "$V"), module $SV)"
[ "$HITS" -gt 0 ] && echo " => the cross-actor fault leak reproduces." \
                  || echo " => did not reproduce this round (weak/marginal — raise N, check perf_env.sh)."
echo "=================================================================="
