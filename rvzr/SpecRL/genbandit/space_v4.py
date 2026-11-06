"""
Factored action space for the Spectre-v4 (speculative store bypass) campaign: one action = one
value per generator knob, written as a config file for an unmodified `revizor.py fuzz` run.

The fixed part is SpecRL's hierarchical_testing/v4.yaml: CT-SEQ, a single basic block and no
conditional branches (so branch misprediction cannot cause a violation and every violation is a
store-bypass one), SSBD off, Unicorn model, with random seeds instead of its fixed ones.
"""
import hashlib
import json
import os
from typing import Dict, List

from space_v1 import ALIASES, GPR_ORDER, _default_register_blocklist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
V4_DIR = os.path.join(HERE, "v4")
SPEC = os.path.join(ROOT, "base.json")

KNOBS: Dict[str, List] = {
    "program_size": [8, 20, 40, 64],
    "mem_ratio": [0.25, 0.5, 0.75],          # avg_mem_accesses = round(mem_ratio * program_size)
    "n_regs": [2, 4, 6],                     # GPR pool, from the front of GPR_ORDER
    "entropy_bits": [4, 10, 16, 24],
    "n_inputs": [10, 25, 50],                # -i of `revizor.py fuzz`, not a config option
    "value_mix": ["default", "full64"],
    "binary": ["on", "off"],                 # BASE-BINARY (add/sub/.../mul/div)
    "flags": ["on", "off"],                  # BASE-CMOV + BASE-SETCC
    "semaphore": ["on", "off"],              # BASE-SEMAPHORE (xadd/cmpxchg)
    "locks": ["on", "off"],                  # off = blocklist every lock-prefixed form
    "spec_filter": ["on", "off"],            # enable_speculation_filter
}

# BASE-DATAXFER (the store and the bypassing load), BASE-LOGICAL and BASE-MISC (lea) are always on
ALWAYS = ["BASE-DATAXFER", "BASE-LOGICAL", "BASE-MISC"]
OPTIONAL = {"binary": ["BASE-BINARY"], "flags": ["BASE-CMOV", "BASE-SETCC"],
            "semaphore": ["BASE-SEMAPHORE"]}

V4_FIXED = """\
instruction_set: x86-64
contract_observation_clause: ct
contract_execution_clause:
  - seq
min_bb_per_function: 1
max_bb_per_function: 1
min_successors_per_bb: 1
max_successors_per_bb: 1
executor_mode: P+P
x86_executor_enable_ssbp_patch: false
inputs_per_class: 2
enable_observation_filter: true
enable_fast_path_model: true
logging_modes: [info]
program_generator_seed: 0
data_generator_seed: 0
"""

# hierarchical_testing/v4.yaml's knobs, i.e. what the control arm runs (with -i 50, as its header says)
CONTROL = {"program_size": 20, "mem_ratio": 0.5, "n_regs": 6, "entropy_bits": 10, "n_inputs": 50,
           "value_mix": "default", "binary": "on", "flags": "on", "semaphore": "on", "locks": "on",
           "spec_filter": "on"}


def _lock_names() -> List[str]:
    return sorted({s["name"] for s in json.load(open(SPEC)) if s["name"].startswith("lock ")})


def action_name(a: Dict) -> str:
    key = ",".join(f"{k}={a[k]}" for k in KNOBS)
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def categories(a: Dict) -> List[str]:
    return ALWAYS + [c for k, cs in OPTIONAL.items() if a[k] == "on" for c in cs]


def register_blocklist(a: Dict) -> List[str]:
    """Full register_blocklist for the action's GPR pool (a yaml list REPLACES the default)."""
    dropped = GPR_ORDER[a["n_regs"]:]
    return _default_register_blocklist() + [x for r in dropped for x in ALIASES[r]]


def to_yaml(a: Dict) -> str:
    size = int(a["program_size"])
    cats = categories(a)
    lines = [V4_FIXED.rstrip(),
             f"program_size: {size}",
             f"avg_mem_accesses: {max(1, round(a['mem_ratio'] * size))}",
             f"data_generator_entropy_bits: {a['entropy_bits']}",
             f"input_value_mix: {a['value_mix']}",
             f"enable_speculation_filter: {'true' if a['spec_filter'] == 'on' else 'false'}",
             "instruction_categories:"] + [f"  - {c}" for c in cats]
    if a["locks"] == "off":
        lines.append("instruction_blocklist_append:")
        lines += [f"  - {n}" for n in _lock_names()]
    if GPR_ORDER[a["n_regs"]:]:
        lines.append("register_blocklist:")
        lines += [f"  - {r}" for r in register_blocklist(a)]
    return "\n".join(lines) + "\n"


def write(a: Dict) -> str:
    os.makedirs(V4_DIR, exist_ok=True)
    path = os.path.join(V4_DIR, f"{action_name(a)}.yaml")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("# " + ", ".join(f"{k}={a[k]}" for k in KNOBS) + "\n" + to_yaml(a))
    return path


def validate(a: Dict) -> str:
    """Load the action's config as `revizor.py fuzz` would; summarize the instruction set."""
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, ROOT)
    from rvzr.config import CONF
    from rvzr.isa_spec import InstructionSet
    from rvzr.arch.x86.fuzzer import _update_instruction_list
    CONF.load(write(a), V4_DIR)
    _update_instruction_list()
    iset = InstructionSet(SPEC, CONF.instruction_categories)
    names = {s.name for s in iset.instructions}
    regs = sorted({v for s in iset.instructions for o in s.operands if o.type.name == "REG"
                   for v in (o.values or []) if v in sum(ALIASES.values(), [])})
    return (f"{action_name(a)} names={len(names)} jcc={sum(n.startswith('j') for n in names)} "
            f"lock={sum(n.startswith('lock') for n in names)} "
            f"mem-names={sum(any(o.type.name == 'MEM' for o in s.operands) for s in iset.instructions)} "
            f"regs={len(regs)} spec_filter={CONF.enable_speculation_filter}")


if __name__ == "__main__":
    import subprocess
    import sys
    # extremes of every knob + the control + everything optional off
    probes = [CONTROL, {**CONTROL, "binary": "off", "flags": "off", "semaphore": "off", "locks": "off"}]
    for k, vals in KNOBS.items():
        for v in (vals[0], vals[-1]):
            probes.append({**CONTROL, k: v})
    for a in probes:
        code = (f"import sys; sys.path.insert(0, '{HERE}'); import space_v4 as s; "
                f"print(s.validate({a!r}))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        diff = {k: v for k, v in a.items() if CONTROL[k] != v}
        print(f"{str(diff or 'CONTROL'):70s} {out.stdout.strip() or out.stderr.strip().splitlines()[-1]}")
