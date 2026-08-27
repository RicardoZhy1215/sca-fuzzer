#!/usr/bin/env bash
#
# dump_htrace_090802.sh — reproduce the minimized 090802 gadget ONCE and show the
# "Violations detected" HTrace table (report_violations -> violation.full_str()).
#
# That table prints automatically whenever `reproduce` finds a violation (no logging mode needed);
# earlier scripts just hid it with >/dev/null. This runs reproduce a single time with output shown,
# and extracts the table:
#
#     ================================ Violations detected ====================
#                              HTrace                     | ID:34  | ID:84 |
#     ^...........^.......................null........... | 420    | 0     |
#     ^....................................null.......... | 0      | 500   |
#
#   Each row = a distinct HTrace (^ = hot cache line); columns = the two counterexample input IDs;
#   cells = how many of the samples produced that HTrace. Same CTr, different HTr = the leak.
#
# NB: single run + ~9/10 signal => ~10% chance this one run is clean (no table). Just re-run if so.
#
# Usage:
#   bash dump_htrace_090802.sh                                   # program.min.asm
#   PROG=$V/program.min.lfence.asm bash dump_htrace_090802.sh    # a variant (expect NO violation)
#   DBG=1 bash dump_htrace_090802.sh                             # also add dbg_violation model traces
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CONFIG=${CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CONFIG" ] || CONFIG=$V/reproduce.yaml
PROG=${PROG:-$V/program.min.asm}
LOG=${LOG:-$CAMP/htrace-090802.log}
DBG=${DBG:-0}

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# optional: add dbg_violation (prints the per-input model/contract trace under the table)
if [ "$DBG" = "1" ]; then
    DBGCFG=$CAMP/config-htrace-090802.yaml
    awk '{print} /^  - info$/ && !d {print "  - dbg_violation"; d=1}' "$CONFIG" > "$DBGCFG"
    CONFIG=$DBGCFG
fi

echo "=================================================================="
echo " DUMP HTRACE TABLE — 090802 (single reproduce run)"
echo "   program : $PROG"
echo "   config  : $CONFIG"
echo "   inputs  : $(ls "$V"/input_*.bin | wc -l) saved   module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "   log     : $LOG"
echo "=================================================================="
cd "$REPO"

b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
"$PY" revizor.py reproduce -s "$BASE" -c "$CONFIG" -I "$V" -t "$PROG" -i "$V"/input_*.bin 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
if [ "$a" -gt "$b" ]; then
    echo ""; echo "!! hpa_gpa EPT COLLISION during the run — result invalid. Reload: bash $CAMP/reload_until_ok.sh"; exit 3
fi

echo ""
echo "=================================================================="
if grep -q "Violations detected" "$LOG"; then
    echo " HTRACE TABLE (extracted):"
    echo "------------------------------------------------------------------"
    awk '/Violations detected/{p=1} p{print} p&&/^Duration:|Statistics/{exit}' "$LOG"
else
    echo " reproduce exit=$RC — no 'Violations detected' table this run"
    echo " (clean this time; the signal is ~9/10, so just re-run: bash $CAMP/dump_htrace_090802.sh)"
fi
echo "=================================================================="
echo " full output: $LOG"
exit "$RC"
