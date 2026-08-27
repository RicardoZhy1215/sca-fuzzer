#!/usr/bin/env bash
#
# test_verw_gadget_090802.sh — SAFE TSA-L1 confirmation: add a VERW inside the observer gadget.
#
# Instead of patching the executor's VM-entry (which wedges), we insert `verw word ptr [r14]` at the
# TOP of the observer function (vm_1), before its faulting load. VERW clears the per-thread microtag
# (with 0208 microcode), so if the leak is TSA-L1 the observer's transient load finds no victim
# microtag entry to false-complete against -> leak closes. This runs at CPL3 in the guest, GIF on,
# so it CANNOT hang vmrun (unlike the switch-macro patch).
#
#   0/N   -> VERW closes it -> CONFIRMED TSA-L1 (microtag false-completion).
#   ~N/N  -> not TSA-L1 (or CPL3 VERW doesn't clear it / not on 0208).
#   ERR   -> the ISA loader rejected `verw` (not in base.json); tell me and I'll add it to the spec.
#
# Uses the CLEAN (reverted) module. Usage:  bash test_verw_gadget_090802.sh   /   N=20 bash ...
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/reproduce.yaml
CTRL=$V/program.min.asm
VW=$V/program.min.verw.asm
N=${N:-10}

[ -f "$CTRL" ] || { echo "no $CTRL"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded"; exit 2; }

# build the VERW variant: insert `verw word ptr [r14]` right after the observer's landing (vm_1)
awk '
  { print }
  /^\.macro\.landing_h2g\.vm_1:/ { print "verw word ptr [r14]" }
' "$CTRL" > "$VW"
[ "$(grep -c '^verw word ptr' "$VW")" -eq 1 ] || { echo "insert failed (landing_h2g.vm_1 not found)"; exit 2; }

echo "=================================================================="
echo " VERW-IN-GADGET TEST — 090802"
echo "   module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)   ucode: $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)"
echo "   variant: $VW  (verw inserted at top of vm_1)"
echo "------------------------------------------------------------------"
grep -nB1 -A1 '^verw word ptr' "$VW" | sed 's/^/   /'
echo "=================================================================="
cd "$REPO"

reproduce_n () {  # $1 prog $2 label ; sets HITS
    local prog=$1 label=$2 hits=0 rc b a
    echo "--- $label : x$N ($(basename "$prog")) ---"
    for r in $(seq 1 "$N"); do
        b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        "$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$prog" -i "$V"/input_*.bin >/dev/null 2>&1
        rc=$?
        a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        if [ "$a" -gt "$b" ]; then echo "  !! hpa_gpa EPT COLLISION — reload_until_ok.sh"; exit 3; fi
        [ "$rc" -eq 1 ] && hits=$((hits+1))
        printf "  run %2d: exit=%d  %s\n" "$r" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    done
    HITS=$hits; echo "  => $label $hits/$N"; echo ""
}

reproduce_n "$CTRL" "CONTROL (no verw)"; CTRL_HITS=$HITS
reproduce_n "$VW"   "VERW    (in vm_1)"; VW_HITS=$HITS

echo "=================================================================="
echo " RESULT:  CONTROL $CTRL_HITS/$N    VERW $VW_HITS/$N"
if [ "$CTRL_HITS" -gt 0 ] && [ "$VW_HITS" -eq 0 ]; then
    echo " => VERW closes it -> CONFIRMED TSA-L1 (microtag false-completion). Case closed."
elif [ "$CTRL_HITS" -gt 0 ] && [ "$VW_HITS" -gt 0 ]; then
    echo " => survives VERW -> not TSA-L1, or CPL3 VERW doesn't clear the microtag (try the switch"
    echo "    patch placed correctly, or confirm ucode 0208)."
else
    echo " => CONTROL didn't reproduce ($CTRL_HITS/$N) — signal down; verify_min_090802.sh first."
fi
echo "=================================================================="