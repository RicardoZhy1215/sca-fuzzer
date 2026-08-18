#!/usr/bin/env bash
#
# verify_alias.sh — decisive check that the host<->host data alias (set_host_data_alias) works.
#
# The model has no notion of the alias: it predicts the observer reads its OWN data. So if the alias
# is live, the observer's HARDWARE registers (derived from [r14+0x100] = the victim's data via the
# alias) diverge from the MODEL's. We count that divergence over a handful of inputs:
#
#   mismatch > 0  -> alias WORKS  (observer sees the victim's data).
#   mismatch == 0 -> alias BROKEN (observer still reads its own data).
#
# Must run with NO fuzzing campaign in flight (the executor is a single-instance resource).
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
CAMP=$REPO/campaigns/vmscape-repro
PY=${PY:-/home/mluo/miniconda3/bin/python3}
KO=$REPO/rvzr/executor_km/rvzr_executor.ko
WANT=$(modinfo -F srcversion "$KO" 2>/dev/null || echo unknown)

# Guard: refuse to touch the executor while a campaign is running.
if pgrep -af "revizor.py" | grep -qv verify_alias; then
    echo "[verify] a revizor campaign appears to be running — refusing (executor is single-instance)."
    exit 3
fi
REF=$(lsmod | awk '/^rvzr_executor/{print $3}')
if [ "${REF:-0}" != "0" ]; then
    echo "[verify] rvzr_executor refcount=$REF (in use) — refusing."; exit 3
fi

LOADED=$(cat /sys/module/rvzr_executor/srcversion 2>/dev/null || echo none)
echo "[verify] loaded=$LOADED want=$WANT"
if [ "$LOADED" != "$WANT" ]; then
    echo "[verify] reloading the fixed build (sudo)"
    lsmod | grep -qw rvzr_executor && sudo rmmod rvzr_executor
    sudo insmod "$KO" || { echo "[verify] insmod FAILED"; exit 2; }
    echo "[verify] now loaded=$(cat /sys/module/rvzr_executor/srcversion)"
fi

cd "$REPO"
PYTHONPATH="$REPO" "$PY" - <<'EOF' 2>&1 | grep -v pkg_resources | tail -12
from rvzr.config import CONF
from rvzr import factory
from rvzr.isa_spec import InstructionSet
CONF.load("campaigns/vmscape-repro/config-k2u-bti.yaml", include_dir="campaigns/vmscape-repro")
iset=InstructionSet("base.json", CONF.instruction_categories)
gen=factory.get_program_generator(CONF.program_generator_seed, iset)
tc=gen.create_test_case_from_template("campaigns/vmscape-repro/template-k2u-poscontrol.asm")
dg=factory.get_data_generator(12345); inputs=dg.generate(8, tc.n_actors())
ex=factory.get_executor(enable_mismatch_check_mode=True); bases=ex.read_base_addresses()
ex.load_test_case(tc); ht=ex.trace_test_case(inputs,1)
m=factory.get_model(bases, enable_mismatch_check_mode=True); m.load_test_case(tc)
ct=m.trace_test_case(inputs, getattr(CONF,"model_max_nesting",1) or 1)
REG=["rax","rbx","rcx","rdx","rsi","rdi"]; nmis=0
for i in range(len(inputs)):
    hw=[int(v) for v in ht[i].get_raw_readings()[0]][:6]; mo=[v%(2**64) for v in ct[i].get_untyped()[:6]]
    if hw!=mo:
        nmis+=1
        if nmis<=2: print(f"  input[{i}]:", [f"{REG[k]}: model={hex(mo[k])} hw={hex(hw[k])}" for k in range(6) if hw[k]!=mo[k]])
print(f"=> {nmis}/{len(inputs)} inputs diverge  ==>  {'ALIAS WORKS' if nmis else 'ALIAS BROKEN'}")
EOF