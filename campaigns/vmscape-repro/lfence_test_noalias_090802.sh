#!/usr/bin/env bash
#
# lfence_test_noalias_090802.sh — is the residual (aliasing-OFF) 090802 leak still transient?
#
# We already know: with hpa_gpa ON the leak is fault-driven + speculative (permission-flip + lfence),
# and with hpa_gpa OFF it still reproduces (~3/5) via a different path (shared L1D sets, no physical
# aliasing). This checks whether that residual is STILL transient by adding an lfence between the
# faulting loads and the retrieval gadget — with hpa_gpa=false.
#
#   CONTROL = program.min.asm         + hpa_gpa=false   -> expect ~3/5 (the residual)
#   LFENCE  = program.min.lfence.asm  + hpa_gpa=false   -> the test
#
# Interpretation:
#   control >0 + lfence 0   -> residual IS transient (lfence kills it) -> fault-driven transient leak
#                              that does NOT need the manufactured aliasing. Stronger result.
#   control >0 + lfence >0  -> residual survives serialization -> it's architectural (P+P through the
#                              shared sets / overlapping window), NOT a transient leak.
#
# NB: the residual is weak (~3/5) and SMT is on after the reboot; run perf_env.sh first for a clean
# read, or use N=10 so a 0-vs-nonzero split is unambiguous.
#
# Usage:
#   bash lfence_test_noalias_090802.sh          # N=5 each
#   N=10 bash lfence_test_noalias_090802.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
SRC_CONFIG=${SRC_CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$SRC_CONFIG" ] || SRC_CONFIG=$V/reproduce.yaml
NOALIAS_CONFIG=$CAMP/config-090802-noalias.yaml
CTRL=$V/program.min.asm
LF=$V/program.min.lfence.asm
N=${N:-5}

[ -f "$CTRL" ] || { echo "no minimized program at $CTRL"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# derive the no-aliasing config (hpa_gpa=false, nothing else changed)
sed 's/^x86_enable_hpa_gpa_collisions:.*/x86_enable_hpa_gpa_collisions: false/' "$SRC_CONFIG" > "$NOALIAS_CONFIG"
grep -q "x86_enable_hpa_gpa_collisions: false" "$NOALIAS_CONFIG" || { echo "sed failed on hpa_gpa key"; exit 2; }

# (re)build the lfence variant: add an lfence right after the post-faulty-load mfence (between the
# transient source and the retrieval/encode gadget)
awk '
  { print }
  /^mfence[[:space:]]+# noremove/ { print "lfence  # SPECULATION BARRIER (lfence-test)" }
' "$CTRL" > "$LF"
[ "$(grep -c 'SPECULATION BARRIER' "$LF")" -eq 1 ] || { echo "transform failed: mfence line not found in $CTRL"; exit 2; }

echo "=================================================================="
echo " LFENCE TEST (aliasing OFF) — 090802"
echo "   control : $CTRL"
echo "   lfence  : $LF"
echo "   config  : $NOALIAS_CONFIG   (x86_enable_hpa_gpa_collisions: false)"
echo "   inputs  : $(ls "$V"/input_*.bin|wc -l)   module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)   microcode: $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)"
echo "   env     : SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null) gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)  (want off/performance; perf_env.sh)"
echo "=================================================================="
cd "$REPO"

reproduce_n () {   # $1 program, $2 label ; sets HITS
    local prog=$1 label=$2 hits=0 rc
    echo "--- $label : reproduce x$N  ($(basename "$prog")) ---"
    for r in $(seq 1 "$N"); do
        "$PY" revizor.py reproduce -s "$BASE" -c "$NOALIAS_CONFIG" -I "$V" -t "$prog" -i "$V"/input_*.bin >/dev/null 2>&1
        rc=$?
        [ "$rc" -eq 1 ] && hits=$((hits+1))
        printf "  run %d: exit=%d  %s\n" "$r" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    done
    HITS=$hits; echo "  => $label reproduced $hits/$N"; echo ""
}

reproduce_n "$CTRL" "CONTROL (no lfence)  "; CTRL_HITS=$HITS
reproduce_n "$LF"   "LFENCE  (spec blocked)"; LF_HITS=$HITS

echo "=================================================================="
echo " RESULT (hpa_gpa OFF):  CONTROL $CTRL_HITS/$N    LFENCE $LF_HITS/$N"
if [ "$CTRL_HITS" -gt 0 ] && [ "$LF_HITS" -eq 0 ]; then
    echo " => residual IS TRANSIENT: lfence kills it even with aliasing off. Fault-driven transient"
    echo "    cross-actor leak that does NOT need the manufactured GPA=HPA collision. Stronger result."
elif [ "$CTRL_HITS" -gt 0 ] && [ "$LF_HITS" -gt 0 ]; then
    echo " => residual SURVIVES serialization -> architectural (P+P via shared L1D sets / overlapping"
    echo "    window), NOT a transient leak. The aliasing-off signal is a different, known channel."
else
    echo " => CONTROL did not reproduce ($CTRL_HITS/$N) with aliasing off this run — signal too weak to"
    echo "    judge (SMT noise). Run perf_env.sh and/or N=10, then re-run."
fi
echo "=================================================================="