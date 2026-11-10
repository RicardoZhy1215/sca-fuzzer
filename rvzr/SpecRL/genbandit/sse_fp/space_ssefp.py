"""
Config space for the SSE floating-point campaign. It has the same interface as space_v4, so
tcenv.py --space ssefp uses it.

The fixed part is hierarchical_testing/sse-fp.yaml, which is side-channel-fuzzer's spec-ext
campaign (sse_fp/README section 8): CT-SEQ, one basic block (no conditional branch, so no
Spectre v1), SSBD on (no Spectre v4), BASE + SSE categories, and the blocked FP arithmetic,
conversions and SSE4.1 FP ops allowlisted by name. Seeds are random instead of fixed. The
speculation filter is OFF: its PFC predicate never fires for FP assists.

Knobs: everything is fixed except entropy_bits, which picks the input arm: 31 = normal values
(the VVI / no-denormal arm), 16 = every float32 input is denormal.
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

# single-valued knobs are fixed; tcenv needs every v4 knob name to exist
KNOBS: Dict[str, List] = {
    "program_size": [64],
    "mem_ratio": [0.375],                    # avg_mem_accesses 24, as sse-fp.yaml / spec-common.yaml
    "n_regs": [6],
    "entropy_bits": [31, 16],                # 31 = normal-value arm, 16 = denormal arm
    "n_inputs": [50],
    "value_mix": ["default"],
    "binary": ["on"],
    "flags": ["on"],
    "semaphore": ["off"],                    # not in sse-fp.yaml's categories
    "locks": ["on"],
    "spec_filter": ["off"],                  # must stay off for FP (see the module docstring)
}

CATEGORIES = ["BASE-BINARY", "BASE-CMOV", "BASE-DATAXFER", "BASE-LOGICAL", "BASE-MISC",
              "BASE-SETCC", "SSE-SSE", "SSE-DATAXFER", "SSE-MISC", "SSE-LOGICAL_FP", "SSE2-SSE",
              "SSE2-DATAXFER", "SSE2-MISC", "SSE2-LOGICAL_FP", "SSE2-LOGICAL", "SSE3-SSE",
              "SSE3-DATAXFER", "SSE4-LOGICAL"]
# side-channel-fuzzer sse_fp/ext-allow.yaml (no cmp*: QEMU rejects predicates >= 8; no dpps/dppd:
# Unicorn rounds them differently; no rsqrt*: an approximation that differs from Unicorn)
ALLOWLIST = ["divps", "divss", "divpd", "divsd", "mulss", "mulps", "mulpd", "mulsd", "sqrtps",
             "sqrtss", "sqrtpd", "sqrtsd", "addps", "addss", "addpd", "addsd", "subps", "subss",
             "subpd", "subsd", "addsubpd", "addsubps", "haddpd", "haddps", "hsubpd", "hsubps",
             "cvtsi2ss", "cvtss2si", "cvttss2si", "cvtsi2sd", "cvtsd2si", "cvttsd2si", "cvtss2sd",
             "cvtsd2ss", "cvtdq2pd", "cvtdq2ps", "cvtpd2dq", "cvttpd2dq", "cvtps2dq", "cvttps2dq",
             "cvtpd2ps", "cvtps2pd", "roundss", "roundsd", "roundps", "roundpd", "blendps",
             "blendpd", "blendvps", "blendvpd", "insertps", "extractps"]

CONTROL = {"program_size": 64, "mem_ratio": 0.375, "n_regs": 6, "entropy_bits": 31, "n_inputs": 50,
           "value_mix": "default", "binary": "on", "flags": "on", "semaphore": "off", "locks": "on",
           "spec_filter": "off"}

FIXED = """\
instruction_set: x86-64
contract_observation_clause: ct
contract_execution_clause:
  - seq
min_bb_per_function: 1
max_bb_per_function: 1
min_successors_per_bb: 1
max_successors_per_bb: 1
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
    return "ssefp-" + hashlib.sha1(key.encode()).hexdigest()[:10]


def categories(a: Dict) -> List[str]:
    return list(CATEGORIES)


def _lock_names() -> List[str]:
    return space_v4._lock_names()


def register_blocklist(a: Dict) -> List[str]:
    dropped = GPR_ORDER[a["n_regs"]:]
    return _default_register_blocklist() + [x for r in dropped for x in ALIASES[r]]


def to_yaml(a: Dict) -> str:
    size = int(a["program_size"])
    lines = [FIXED.rstrip(),
             f"program_size: {size}",
             f"avg_mem_accesses: {max(1, round(a['mem_ratio'] * size))}",
             f"data_generator_entropy_bits: {a['entropy_bits']}",
             f"input_value_mix: {a['value_mix']}",
             f"enable_speculation_filter: {'true' if a['spec_filter'] == 'on' else 'false'}",
             "instruction_categories:"] + [f"  - {c}" for c in categories(a)]
    lines.append("instruction_allowlist:")
    lines += [f"  - {n}" for n in ALLOWLIST]
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
