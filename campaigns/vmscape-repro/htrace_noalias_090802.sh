#!/usr/bin/env bash
#
# htrace_noalias_090802.sh — characterize the aliasing-OFF channel: value-forward (FILL) vs
# contention (EVICT).
#
# Captures the "Violations detected" HTrace table under BOTH configs (retrying until a violation
# prints), extracts the two counterexample htraces, and classifies the differing cache line:
#   FILL  = the differing line is an EXTRA hot line (^ in one input, . in the other, and NOT part of
#           the observer's architectural footprint) -> value-forward (transient store to a value-
#           derived address). This is what the aliased case does (line 12).
#   EVICT = the differing line IS one of the observer's normally-hot lines, and it goes COLD in one
#           input -> the victim evicted a shared L1D set -> contention.
#
# The observer's architectural footprint is taken from the ALIASED run's common lines (the lines hot
# for both inputs); the aliasing-OFF differing line is classified against it.
#
# Usage:
#   bash htrace_noalias_090802.sh              # K=8 retries per config
#   K=15 bash htrace_noalias_090802.sh         # more retries (weak/noisy signal)
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CTRL_CONFIG=${CTRL_CONFIG:-$CAMP/config-refuzz-u2k.yaml}   # aliased (hpa_gpa=true)
[ -f "$CTRL_CONFIG" ] || CTRL_CONFIG=$V/reproduce.yaml
NOALIAS_CONFIG=$CAMP/config-090802-noalias.yaml
PROG=${PROG:-$V/program.min.asm}
K=${K:-8}
ALOG=$V/htrace-aliased.log
NLOG=$V/htrace-noalias.log

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded"; exit 2; }

sed 's/^x86_enable_hpa_gpa_collisions:.*/x86_enable_hpa_gpa_collisions: false/' "$CTRL_CONFIG" > "$NOALIAS_CONFIG"
grep -q "x86_enable_hpa_gpa_collisions: false" "$NOALIAS_CONFIG" || { echo "sed failed on hpa_gpa key"; exit 2; }

cd "$REPO"
capture () {   # $1 config, $2 logfile ; returns 0 if a violation table was captured
    local cfg=$1 log=$2 i
    for i in $(seq 1 "$K"); do
        "$PY" revizor.py reproduce -s "$BASE" -c "$cfg" -I "$V" -t "$PROG" -i "$V"/input_*.bin > "$log" 2>&1
        grep -q "Violations detected" "$log" && return 0
    done
    return 1
}

echo "=================================================================="
echo " HTRACE CHARACTERIZATION — 090802 (FILL vs EVICT), aliasing on vs off"
echo "   program : $PROG   module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)   ucode: $(awk '/^microcode/{print $3; exit}' /proc/cpuinfo)"
echo "   SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null) gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)  (perf_env.sh for a clean read)"
echo "=================================================================="
echo "[capture] aliased (hpa_gpa=true) — up to $K tries..."
capture "$CTRL_CONFIG" "$ALOG"    && echo "  got a violation table" || echo "  NO violation in $K tries (aliased)"
echo "[capture] no-alias (hpa_gpa=false) — up to $K tries..."
capture "$NOALIAS_CONFIG" "$NLOG" && echo "  got a violation table" || echo "  NO violation in $K tries (no-alias)"

echo ""
"$PY" - "$ALOG" "$NLOG" <<'EOF'
import re, sys
def parse(path):
    rows=[]
    try: lines=open(path).read().splitlines()
    except: return None
    for ln in lines:
        m=re.match(r'\s*([\^.]{8,})\s*\|\s*(\d+)\s*\|\s*(\d+)', ln)
        if m: rows.append((m.group(1), int(m.group(2)), int(m.group(3))))
    if not rows: return None
    # representative htrace per input = the row with the max count in that column
    rA=max(rows, key=lambda r:r[1]); rB=max(rows, key=lambda r:r[2])
    return rA[0], rB[0]
def hot(s): return {i for i,c in enumerate(s) if c=='^'}

al=parse(sys.argv[1]); na=parse(sys.argv[2])
print("================================================================")
if al:
    a,b=al; ha,hb=hot(a),hot(b)
    print(f"ALIASED  input-A hot lines: {sorted(ha)}")
    print(f"         input-B hot lines: {sorted(hb)}")
    print(f"         common (footprint): {sorted(ha&hb)}   differing: {sorted(ha^hb)}")
else:
    print("ALIASED  : no table captured")
print("----------------------------------------------------------------")
if na:
    a,b=na; ha,hb=hot(a),hot(b)
    common=ha&hb; diff=ha^hb
    print(f"NO-ALIAS input-A hot lines: {sorted(ha)}")
    print(f"         input-B hot lines: {sorted(hb)}")
    print(f"         common (footprint): {sorted(common)}   differing: {sorted(diff)}")
    # footprint reference = aliased common lines (the observer's always-present architectural accesses)
    ref = (hot(al[0]) & hot(al[1])) if al else common
    print("----------------------------------------------------------------")
    print(f" footprint reference (aliased common): {sorted(ref)}")
    for d in sorted(diff):
        kind = "EVICT (contention — a footprint line goes cold)" if d in ref else "FILL (value-forward — an extra line goes hot)"
        who = "A" if d in ha else "B"
        print(f"   differing line {d}: hot only in input {who}  ->  {kind}")
    if not diff:
        print("   (no differing line — capture noisy; re-run with perf_env + higher K)")
else:
    print("NO-ALIAS : no table captured (raise K, run perf_env.sh)")
print("================================================================")
print(" verdict: if the aliasing-OFF differing line is FILL -> value is being transiently")
print("          forwarded even without physical aliasing (stronger). If EVICT -> it's transient")
print("          set-contention (attacker senses which set the victim touched).")
print(" full tables: $ALOG / $NLOG")
EOF
echo " (raw tables: $ALOG  and  $NLOG)"