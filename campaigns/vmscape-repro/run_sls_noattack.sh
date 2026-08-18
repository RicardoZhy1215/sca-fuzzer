#!/usr/bin/env bash
# run_sls_noattack.sh — the "SLS barrier + no attacker" test.
# SLS is blocked by the lfence fall-through, and there is NO attacker BTB training. So neither
# straight-line speculation nor injection can reach the disclosure. P+P channel, no shared memory.
#   >0 violations => the leak is NEITHER SLS nor injection => a generic transient in the victim's
#                    random preamble (vmv_pre). (Later pinned by inject2-nopre, which removes it -> 0.)
# Does NOT touch the kernel module — load the clean 1CD139EF yourself first.
# Usage:  bash run_sls_noattack.sh     |     N=500 bash run_sls_noattack.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer; CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}; BASE=$REPO/base.json
CONFIG=$CAMP/config-v2v-pp.yaml
TEMPLATE=$CAMP/template-v2v-sls-noattack.asm
OUT=$CAMP/out-v2v-sls-noattack
N=${N:-200}; I=${I:-50}
echo "=================================================================="
echo " SLS barrier + NO attacker   (P+P, no shared memory)   N=$N x I=$I"
echo " template: $TEMPLATE"
echo "=================================================================="
echo "loaded executor: $(cat /sys/module/rvzr_executor/srcversion 2>/dev/null)  (want clean 1CD139EF)"
[ -e /sys/rvzr_executor ] || { echo "no /sys/rvzr_executor — load the module first"; exit 2; }
mkdir -p "$OUT"; cd "$REPO"
"$PY" revizor.py tfuzz -s "$BASE" -c "$CONFIG" -t "$TEMPLATE" -n "$N" -i "$I" -w "$OUT"
shopt -s nullglob; V=("$OUT"/violation-*)
echo "=================================================================="
echo " SLS barrier + NO attacker => violations: ${#V[@]}"
if [ "${#V[@]}" -gt 0 ]; then
  echo "  => leak is NEITHER SLS nor injection => generic transient in the victim's random preamble"
else
  echo "  => clean (unexpected here — the earlier N=100 run found 1)"
fi
echo "=================================================================="
