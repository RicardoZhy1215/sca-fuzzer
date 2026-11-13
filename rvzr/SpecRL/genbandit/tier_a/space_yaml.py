"""
A config space backed by one Revizor yaml file, for tcenv.py --space yaml --space-yaml <file>.

Every knob is fixed to the value in the yaml, so the policy only writes the programs (as ctxrl does in
all campaigns); the contract, faults, filters, executor settings, instruction lists etc. come from the
yaml unchanged. Same interface as space_v4 / space_v1ctx / space_ssefp (KNOBS, CONTROL, SPEC,
categories, register_blocklist, _lock_names, write). Seeds should be 0 (random) in the yaml.

  import space_yaml as space; space.configure("tier_a/configs/sco.yaml")
"""
import hashlib
import os
import sys
from typing import Dict, List

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GB = os.path.dirname(HERE)                               # genbandit/
sys.path.insert(0, GB)
from space_v1 import _default_register_blocklist  # noqa: E402
import space_v4  # noqa: E402

ROOT = space_v4.ROOT
SPEC = space_v4.SPEC
CFG_DIR = os.path.join(HERE, "configs", "derived")

SRC = None          # the yaml file
TEXT = ""
Y: Dict = {}
KNOBS: Dict[str, List] = {}
CONTROL: Dict = {}


def configure(path: str) -> None:
    global SRC, TEXT, Y, KNOBS, CONTROL
    SRC = os.path.abspath(path)
    TEXT = open(SRC).read()
    Y = yaml.safe_load(TEXT) or {}
    assert "instruction_categories" in Y, "the yaml must list instruction_categories"
    assert not Y.get("register_blocklist"), "register_blocklist in the yaml is not supported here"
    size = int(Y.get("program_size", 24))
    mem = int(Y.get("avg_mem_accesses", 12))
    CONTROL = {"program_size": size, "mem_ratio": mem / size, "n_regs": 6,
               "entropy_bits": int(Y.get("data_generator_entropy_bits", 31)), "n_inputs": 50,
               "value_mix": Y.get("input_value_mix", "default"), "binary": "on", "flags": "on",
               "semaphore": "on", "locks": "on",
               "spec_filter": "on" if Y.get("enable_speculation_filter", True) else "off"}
    KNOBS = {k: [v] for k, v in CONTROL.items()}            # single-valued: fixed by the yaml


def action_name(a: Dict) -> str:
    base = os.path.splitext(os.path.basename(SRC))[0]
    return base + "-" + hashlib.sha1(TEXT.encode()).hexdigest()[:10]


def categories(a: Dict) -> List[str]:
    return list(Y["instruction_categories"])


def _lock_names() -> List[str]:
    return space_v4._lock_names()


def register_blocklist(a: Dict) -> List[str]:
    return _default_register_blocklist()


def write(a: Dict) -> str:
    """the yaml itself (copied, so a run records exactly what it used)"""
    assert SRC is not None, "call configure(path) first"
    os.makedirs(CFG_DIR, exist_ok=True)
    path = os.path.join(CFG_DIR, f"{action_name(a)}.yaml")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(f"# copied from {os.path.relpath(SRC, GB)}\n" + TEXT)
    return path
