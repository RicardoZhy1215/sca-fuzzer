#!/usr/bin/env bash
#
# privilege_ab_090802.sh — decisive A/B: does the KNOWN 40/40 gadget (violation-260817-090802) leak
# because of the guest-USER observer (user/supervisor fault), or independently of privilege?
#
# Reproduces the SAME program + SAME inputs under two configs that differ in ONE thing:
#   CONTROL : vm (observer) = guest USER    (config-refuzz-u2k.yaml, as found)   -> expect ~10/10
#   KERNEL  : vm (observer) = guest KERNEL   (both guests kernel, like the paper) -> the test
#
# No fuzzing-depth confound (unlike a from-scratch both-kernel campaign): it's the exact program we
# already know leaks 10/10 at user, just re-run with the observer's privilege flipped.
#
# Interpretation:
#   user 10/10 + kernel 0/10  -> the leak DEPENDS on the user/supervisor fault; it is NOT the
#                                both-kernel TSA-L1D. (And since meltdown=Not affected, that user-fault
#                                transient is either a distinct AMD effect or a guest-user sandbox
#                                artifact — needs scrutiny.)
#   user 10/10 + kernel 10/10 -> privilege is irrelevant; the baseline's null was just fuzzing depth.
#
# Usage:
#   bash privilege_ab_090802.sh              # N=10 each, on program.asm
#   N=20 bash privilege_ab_090802.sh
#   PROG=$V/program.min.asm bash privilege_ab_090802.sh   # use the minimized gadget instead
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
BASE=$REPO/base.json
V=${V:-$CAMP/out-refuzz-u2k/violation-260817-090802}
CTRL_CONFIG=${CTRL_CONFIG:-$CAMP/config-refuzz-u2k.yaml}
[ -f "$CTRL_CONFIG" ] || CTRL_CONFIG=$V/reproduce.yaml
KERN_CONFIG=$CAMP/config-090802-bothkernel.yaml
PROG=${PROG:-$V/program.asm}
N=${N:-10}

[ -f "$PROG" ] || { echo "no program at $PROG"; exit 2; }
[ -e /sys/rvzr_executor ] || { echo "module not loaded — sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"; exit 2; }

# derive the both-kernel config: flip the single observer 'user' -> 'kernel', change nothing else
sed 's/privilege_level: "user"/privilege_level: "kernel"/' "$CTRL_CONFIG" > "$KERN_CONFIG"
uc=$(grep -cE 'privilege_level.*user' "$KERN_CONFIG")
[ "$uc" -eq 0 ] || { echo "sed failed — kernel config still has $uc user line(s)"; exit 2; }

echo "=================================================================="
echo " PRIVILEGE A/B — violation-260817-090802  (same program+inputs, flip observer privilege)"
echo "   program : $PROG  ($(grep -vE '^\s*(#|\.|$)' "$PROG" | grep -cvE '^\s*$') instrs)"
echo "   inputs  : $(ls "$V"/input_*.bin | wc -l) saved   module: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)"
echo "   CONTROL : vm=USER    ($CTRL_CONFIG)"
echo "   KERNEL  : vm=KERNEL  ($KERN_CONFIG, both guests kernel)"
echo "=================================================================="
cd "$REPO"

reproduce_n () {   # $1 program, $2 config, $3 label ; sets HITS
    local prog=$1 cfg=$2 label=$3 hits=0 rc b a
    echo "--- $label : reproduce x$N ---"
    for i in $(seq 1 "$N"); do
        b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        "$PY" revizor.py reproduce -s "$BASE" -c "$cfg" -I "$V" -t "$prog" -i "$V"/input_*.bin >/dev/null 2>&1
        rc=$?
        a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
        if [ "$a" -gt "$b" ]; then
            echo "  !! hpa_gpa EPT COLLISION — reload: bash $CAMP/reload_until_ok.sh"; exit 3
        fi
        [ "$rc" -eq 1 ] && hits=$((hits+1))
        printf "  run %2d: exit=%d  %s\n" "$i" "$rc" "$([ "$rc" -eq 1 ] && echo VIOLATION || { [ "$rc" -eq 0 ] && echo clean || echo ERR; })"
    done
    HITS=$hits
    echo "  => $label reproduced $hits/$N"; echo ""
}

reproduce_n "$PROG" "$CTRL_CONFIG" "CONTROL (vm=USER)";   USER_HITS=$HITS
reproduce_n "$PROG" "$KERN_CONFIG" "KERNEL  (vm=KERNEL)"; KERN_HITS=$HITS

echo "=================================================================="
echo " RESULT:  USER $USER_HITS/$N    KERNEL $KERN_HITS/$N"
if [ "$USER_HITS" -gt 0 ] && [ "$KERN_HITS" -eq 0 ]; then
    echo " => leak DEPENDS on the guest-USER privilege (user/supervisor fault). NOT both-kernel TSA-L1D."
    echo "    Next: characterize the fault (is meltdown-like on a CPU that reports meltdown=Not affected?),"
    echo "    and rule out a Revizor guest-user sandbox artifact."
elif [ "$USER_HITS" -gt 0 ] && [ "$KERN_HITS" -gt 0 ]; then
    echo " => privilege is IRRELEVANT — it leaks at kernel too. The baseline null was fuzzing depth."
    echo "    Then this IS consistent with both-kernel TSA-L1D; re-run baseline with larger N to confirm."
else
    echo " => CONTROL did not reproduce ($USER_HITS/$N) — signal down this session. Confirm first:"
    echo "    bash $CAMP/replay_refuzz_u2k.sh   then re-run this A/B."
fi
echo "=================================================================="
