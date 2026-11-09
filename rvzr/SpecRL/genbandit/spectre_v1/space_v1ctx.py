"""
Config space for the Spectre-v1 (bounds-check bypass) campaign with ctxrl. It has the same
interface as space_v4 (KNOBS, CONTROL, SPEC, categories, register_blocklist, _lock_names, write),
so tcenv.py --space v1 can use it in place of space_v4.

The fixed part is SpecRL's v1.yaml: CT-SEQ, both filters on, Unicorn model, and SSBD on
(x86_executor_enable_ssbp_patch), so store bypass cannot cause a violation. Seeds are random
instead of fixed. Several basic blocks are joined by conditional branches (BASE-COND_BR).
Revizor draws the CFG and the jumps; the policy writes only the instructions inside the blocks.

Instruction categories are the v4 ones (DATAXFER, LOGICAL, MISC, and the optional BINARY, CMOV +
SETCC, SEMAPHORE), plus COND_BR, NOP, BITBYTE, CONVERT and FLAGOP. PUSH and POP are left out:
they implicitly write rsp, so Revizor filters out every form of them.
"""
import hashlib
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
GB = os.path.dirname(HERE)                               # genbandit/
sys.path.insert(0, GB)
from space_v1 import ALIASES, GPR_ORDER, _default_register_blocklist  # noqa: E402
import space_v4  # noqa: E402

ROOT = space_v4.ROOT
SPEC = space_v4.SPEC
CFG_DIR = os.path.join(HERE, "configs")

KNOBS: Dict[str, List] = {
    "program_size": [8, 16, 24, 32, 48, 64],
    "mem_ratio": [0.25, 0.5, 0.75],          # avg_mem_accesses = round(mem_ratio * program_size)
    "n_regs": [2, 4, 6],                     # GPR pool, from the front of GPR_ORDER
    "entropy_bits": [4, 10, 16, 24],
    "n_inputs": [10, 25, 50],                # -i of `revizor.py fuzz`, not a config option
    "value_mix": ["default", "full64"],
    "binary": ["on", "off"],                 # BASE-BINARY
    "flags": ["on", "off"],                  # BASE-CMOV + BASE-SETCC
    "semaphore": ["on", "off"],              # BASE-SEMAPHORE
    "locks": ["on", "off"],                  # off = blocklist every lock-prefixed form
    "spec_filter": ["on", "off"],            # enable_speculation_filter
    "n_bb": [1, 2, 3, 5],                    # basic blocks per function (min = max)
    "successors": ["2", "1-2"],              # successors per basic block (2 = every block branches)
}

ALWAYS = ["BASE-DATAXFER", "BASE-LOGICAL", "BASE-MISC", "BASE-COND_BR",
          "BASE-NOP", "BASE-BITBYTE", "BASE-CONVERT", "BASE-FLAGOP"]
OPTIONAL = space_v4.OPTIONAL

# SpecRL v1.yaml's knobs, with every optional category on
CONTROL = {"program_size": 24, "mem_ratio": 0.5, "n_regs": 6, "entropy_bits": 24, "n_inputs": 50,
           "value_mix": "default", "binary": "on", "flags": "on", "semaphore": "on", "locks": "on",
           "spec_filter": "on", "n_bb": 2, "successors": "2"}

FIXED = """\
instruction_set: x86-64
contract_observation_clause: ct
contract_execution_clause:
  - seq
executor_mode: P+P
x86_executor_enable_ssbp_patch: true
inputs_per_class: 2
enable_observation_filter: true
enable_fast_path_model: true
logging_modes: [info]
program_generator_seed: 0
data_generator_seed: 0
"""


def action_name(a: Dict) -> str:
    key = ",".join(f"{k}={a[k]}" for k in KNOBS)
    return "v1-" + hashlib.sha1(key.encode()).hexdigest()[:10]


def categories(a: Dict) -> List[str]:
    return ALWAYS + [c for k, cs in OPTIONAL.items() if a[k] == "on" for c in cs]


def _lock_names() -> List[str]:
    return space_v4._lock_names()


def register_blocklist(a: Dict) -> List[str]:
    """Full register_blocklist for the action's GPR pool (a yaml list REPLACES the default)."""
    dropped = GPR_ORDER[a["n_regs"]:]
    return _default_register_blocklist() + [x for r in dropped for x in ALIASES[r]]


def to_yaml(a: Dict) -> str:
    size = int(a["program_size"])
    lo, hi = (2, 2) if a["successors"] == "2" else (1, 2)
    lines = [FIXED.rstrip(),
             f"min_bb_per_function: {a['n_bb']}",
             f"max_bb_per_function: {a['n_bb']}",
             f"min_successors_per_bb: {lo}",
             f"max_successors_per_bb: {hi}",
             f"program_size: {size}",
             f"avg_mem_accesses: {max(1, round(a['mem_ratio'] * size))}",
             f"data_generator_entropy_bits: {a['entropy_bits']}",
             f"input_value_mix: {a['value_mix']}",
             f"enable_speculation_filter: {'true' if a['spec_filter'] == 'on' else 'false'}",
             "instruction_categories:"] + [f"  - {c}" for c in categories(a)]
    if a["locks"] == "off":
        lines.append("instruction_blocklist_append:")
        lines += [f"  - {n}" for n in _lock_names()]
    if GPR_ORDER[a["n_regs"]:]:
        lines.append("register_blocklist:")
        lines += [f"  - {r}" for r in register_blocklist(a)]
    return "\n".join(lines) + "\n"


def write(a: Dict) -> str:
    os.makedirs(CFG_DIR, exist_ok=True)
    path = os.path.join(CFG_DIR, f"{action_name(a)}.yaml")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("# " + ", ".join(f"{k}={a[k]}" for k in KNOBS) + "\n" + to_yaml(a))
    return path
