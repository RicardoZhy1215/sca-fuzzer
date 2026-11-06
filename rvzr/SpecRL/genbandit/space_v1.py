"""
Factored action space for the Spectre-v1 campaign: one action = one value per generator knob.
Every knob is an existing Revizor config option (or the -i input count), so an action is just
a config file for an unmodified `revizor.py fuzz` run.

The fixed part is SpecRL's v1.yaml (CT-SEQ, BASE categories incl. jcc, both filters on, SSBD,
Unicorn model) with random seeds instead of its fixed ones.
"""
import hashlib
import os
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(HERE, "v1")

KNOBS: Dict[str, List] = {
    "program_size": [8, 16, 24, 48],
    "mem_ratio": [0.25, 0.5, 0.75],          # avg_mem_accesses = round(mem_ratio * program_size)
    "n_bb": [2, 3, 5],
    "successors": ["2", "1-2"],
    "n_regs": [2, 4, 6],                     # GPR pool, from the front of GPR_ORDER
    "entropy_bits": [4, 12, 24, 31],
    "n_inputs": [10, 25, 50],                # -i of `revizor.py fuzz`, not a config option
    "value_mix": ["default", "full64"],
}

GPR_ORDER = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi"]
ALIASES = {"rax": ["rax", "eax", "ax", "al", "ah"], "rbx": ["rbx", "ebx", "bx", "bl", "bh"],
           "rcx": ["rcx", "ecx", "cx", "cl", "ch"], "rdx": ["rdx", "edx", "dx", "dl", "dh"],
           "rsi": ["rsi", "esi", "si", "sil"], "rdi": ["rdi", "edi", "di", "dil"]}

V1_FIXED = """\
instruction_set: x86-64
contract_observation_clause: ct
contract_execution_clause:
  - no_speculation
instruction_categories:
  - BASE-COND_BR
  - BASE-BINARY
  - BASE-CMOV
  - BASE-DATAXFER
  - BASE-LOGICAL
  - BASE-MISC
  - BASE-SEMAPHORE
  - BASE-SETCC
executor_mode: P+P
x86_executor_enable_ssbp_patch: true
inputs_per_class: 2
enable_speculation_filter: true
enable_observation_filter: true
enable_fast_path_model: true
logging_modes: [info]
program_generator_seed: 0
data_generator_seed: 0
"""

# SpecRL's v1.yaml knobs, i.e. what the control arm runs (with -i CONTROL_INPUTS)
CONTROL = {"program_size": 24, "mem_ratio": 0.5, "n_bb": 2, "successors": "2", "n_regs": 6,
           "entropy_bits": 24, "n_inputs": 50, "value_mix": "default"}


def _default_register_blocklist() -> List[str]:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
    from rvzr.arch.x86 import config as xc
    return list(xc.register_blocklist)


def action_name(a: Dict) -> str:
    key = ",".join(f"{k}={a[k]}" for k in KNOBS)
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def to_yaml(a: Dict) -> str:
    size = int(a["program_size"])
    lo, hi = (2, 2) if a["successors"] == "2" else (1, 2)
    lines = [V1_FIXED.rstrip(),
             f"program_size: {size}",
             f"avg_mem_accesses: {max(1, round(a['mem_ratio'] * size))}",
             f"min_bb_per_function: {a['n_bb']}",
             f"max_bb_per_function: {a['n_bb']}",
             f"min_successors_per_bb: {lo}",
             f"max_successors_per_bb: {hi}",
             f"data_generator_entropy_bits: {a['entropy_bits']}",
             f"input_value_mix: {a['value_mix']}"]
    dropped = GPR_ORDER[a["n_regs"]:]
    if dropped:
        # a yaml register_blocklist REPLACES the x86 default list, so write the full list
        full = _default_register_blocklist() + [x for r in dropped for x in ALIASES[r]]
        lines.append("register_blocklist:")
        lines += [f"  - {r}" for r in full]
    return "\n".join(lines) + "\n"


def write(a: Dict) -> str:
    os.makedirs(V1_DIR, exist_ok=True)
    path = os.path.join(V1_DIR, f"{action_name(a)}.yaml")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("# " + ", ".join(f"{k}={a[k]}" for k in KNOBS) + "\n" + to_yaml(a))
    return path


def validate(a: Dict) -> str:
    """Load the action's config as `revizor.py fuzz` would; summarize the instruction set."""
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    root = os.path.abspath(os.path.join(HERE, "../../.."))
    sys.path.insert(0, root)
    from rvzr.config import CONF
    from rvzr.isa_spec import InstructionSet
    from rvzr.arch.x86.fuzzer import _update_instruction_list
    CONF.load(write(a), V1_DIR)
    _update_instruction_list()
    iset = InstructionSet(os.path.join(root, "base.json"), CONF.instruction_categories)
    names = {s.name for s in iset.instructions}
    regs = sorted({v for s in iset.instructions for o in s.operands if o.type.name == "REG"
                   for v in (o.values or []) if v in sum(ALIASES.values(), [])})
    return f"{action_name(a)} names={len(names)} jcc={sum(n.startswith('j') for n in names)} regs={regs}"


if __name__ == "__main__":
    import itertools
    import subprocess
    import sys
    # extremes of every knob + the control
    probes = [CONTROL]
    for k, vals in KNOBS.items():
        for v in (vals[0], vals[-1]):
            probes.append({**CONTROL, k: v})
    for a in probes:
        code = (f"import sys; sys.path.insert(0, '{HERE}'); import space_v1 as s; "
                f"print(s.validate({a!r}))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        diff = {k: v for k, v in a.items() if CONTROL[k] != v}
        print(f"{str(diff or 'CONTROL'):32s} {out.stdout.strip() or out.stderr.strip().splitlines()[-1]}")
