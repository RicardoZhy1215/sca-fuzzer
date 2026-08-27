#!/usr/bin/env bash
#
# test_verw_090802.sh — does adding VERW to the guest-entry switch close the 090802 leak?
#
# Run this AFTER rebuilding + reloading the module with the VERW patch in body_macro_switch_h2g
# (macros.c). The patch clears the microtag before each vmrun — the mitigation the executor's AMD
# uarch_flush stub omits. Requires microcode 0208 (VERW only clears TSA state with that ucode).
#
#   baseline (no VERW): 090802 reproduced ~5/5..10/10.
#   with VERW:
#     0/N   -> leak CLOSED by VERW -> CONFIRMED TSA-L1 (microtag false-completion). Investigation done.
#     ~N/N  -> VERW didn't help -> not TSA-L1 (or wrong clear point).
#
# Usage:  bash test_verw_090802.sh          # N=10
#         N=20 bash test_verw_090802.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/reproduce.yaml
PROG=${PROG:-$V/program.min.asm}
N=${N:-10}

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — rebuild+insmod first (see below)"; exit 2; }

sv=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)
uc=$(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)
echo "=================================================================="
echo " VERW TEST — 090802 (does clearing the microtag on guest-entry close it?)"
echo "   module srcversion : $sv"
[ "$sv" = "1CD139EF2BF9FE9618F4A95" ] && echo "   !! this is the OLD (no-VERW) build — rebuild+reload first, or the test is meaningless."
echo "   microcode : $uc   (need 0xa705208 for VERW to clear TSA state)"
[ "$uc" != "0xa705208" ] && echo "   !! not on 0208 — VERW won't clear the microtag; boot 0208 first."
echo "   program   : $PROG   config: $CONFIG"
echo "=================================================================="
cd "$REPO"
hits=0
for r in $(seq 1 "$N"); do
    b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
    "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$PROG" -i "$V"/input_*.bin >/dev/null 2>&1
    rc=$?
    a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
    if [ "$a" -gt "$b" ]; then echo "  !! hpa_gpa EPT COLLISION — bash $CAMP/reload_until_ok.sh, then retry"; exit 3; fi
    [ "$rc" -eq 1 ] && hits=$((hits+1))
    printf "  run %2d: exit=%d  %s\n" "$r" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
done
echo "=================================================================="
echo " RESULT (VERW build): reproduced $hits/$N   (baseline without VERW was ~5/5..10/10)"
if [ "$hits" -eq 0 ]; then
    echo " => VERW CLOSED the leak -> CONFIRMED TSA-L1 (microtag false-completion). Case closed:"
    echo "    the violation is the known, already-mitigated TSA-L1; it only reproduced because the"
    echo "    executor's AMD uarch_flush is a stub that skips VERW."
else
    echo " => leak survives VERW -> NOT TSA-L1 (or the clear is in the wrong place / ucode not 0208)."
fi
echo "=================================================================="