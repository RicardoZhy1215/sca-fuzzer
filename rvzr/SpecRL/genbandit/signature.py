"""
Class signature of a violation, for "is this a NEW violation class?".

signature = the set of instruction groups the violation NEEDS: for every group present in the
program, delete all of its instructions and re-check; the groups whose deletion kills the
violation form the signature. A group is "<uops.info category>/<fp|int>" (fp = the name has an
f32/f64-typed register operand), so e.g. the known FPVI cases come out as {SSE2-SSE/fp, ...}.

Every check uses the violation's minimize.yaml with -n <inputs>: inputs are regenerated from the
saved seed and RE-BOOSTED with the variant's own taints. Reusing the saved boosted inputs is
wrong for a modified program -- its pairs need not be contract-equivalent any more, so a
violation can "disappear" for reasons unrelated to the deleted group (see memory:
avx-masked-store-violations).
"""
import json
import os
import sys
import re
import subprocess
import time
from typing import Dict, FrozenSet, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
SPEC = os.path.join(ROOT, "base-avx.json")
PY = sys.executable
FP_XTYPES = {"f16", "f32", "f64", "f80", "bf16"}

_GROUP_OF: Dict[str, str] = {}


def _load_groups() -> Dict[str, str]:
    if not _GROUP_OF:
        cats: Dict[str, set] = {}
        fp: Dict[str, bool] = {}
        for i in json.load(open(SPEC)):
            cats.setdefault(i["name"], set()).add(i["category"])
            is_fp = any(o["type_"] == "REG" and o.get("xtype") in FP_XTYPES for o in i["operands"])
            fp[i["name"]] = fp.get(i["name"], False) or is_fp
        for name, cs in cats.items():
            cs = sorted(c for c in cs if c != "BASE-STRINGOP") or sorted(cs)
            _GROUP_OF[name] = f"{cs[0]}/{'fp' if fp[name] else 'int'}"
    return _GROUP_OF


def _mnemonic(line: str) -> Optional[str]:
    s = line.split("#")[0].strip()
    if not s or s.startswith(".") or s.endswith(":"):
        return None
    words = s.split()
    return " ".join(words[:2]) if words[0] == "lock" and len(words) > 1 else words[0]


def instruction_lines(asm: str) -> List[Tuple[int, str, str]]:
    """(line index, mnemonic, group) of every generated, non-instrumentation instruction."""
    groups = _load_groups()
    out = []
    for k, line in enumerate(asm.splitlines()):
        if "instrumentation" in line:
            continue
        m = _mnemonic(line)
        if m is None or m.startswith("j") or m == "nop":
            continue
        g = groups.get(m) or groups.get(m.split()[-1])
        if g:
            out.append((k, m, g))
    return out


class CheckError(RuntimeError):
    """reproduce crashed: neither 'violates' nor 'does not violate' -- never read as either."""


def violates(vdir: str, asm_path: str, n_inputs: int, include_dir: str = HERE,
             timeout: int = 900) -> bool:
    cmd = [PY, "revizor.py", "reproduce", "-s", SPEC, "-c", os.path.join(vdir, "minimize.yaml"),
           "-I", include_dir, "-t", asm_path, "-n", str(n_inputs)]
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise CheckError(f"timeout after {timeout} s: {asm_path}") from e
    if "Violations detected" in p.stdout:
        return True
    if "Traceback" in p.stdout + p.stderr or "Duration:" not in p.stdout:
        tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
        raise CheckError(f"reproduce failed on {asm_path}: {' | '.join(tail)}")
    return False


def signature(vdir: str, n_inputs: int = 50,
              include_dir: str = HERE) -> Tuple[Optional[FrozenSet[str]], dict]:
    """None if the violation does not reproduce at all; else the frozenset of necessary groups
    (possibly empty: no single group is necessary, e.g. two redundant transmitters)."""
    t0 = time.time()
    asm = open(os.path.join(vdir, "program.asm")).read()
    info = {"checks": 0}
    info["checks"] += 1
    if not violates(vdir, os.path.join(vdir, "program.asm"), n_inputs, include_dir):
        info["seconds"] = round(time.time() - t0, 1)
        return None, info
    lines = instruction_lines(asm)
    present = sorted({g for _, _, g in lines})
    needed = []
    src = asm.splitlines()
    for g in present:
        drop = {k for k, _, gg in lines if gg == g}
        variant = os.path.join(vdir, f"sig-without-{re.sub(r'[^A-Za-z0-9]+', '_', g)}.asm")
        open(variant, "w").write("\n".join(l for k, l in enumerate(src) if k not in drop) + "\n")
        info["checks"] += 1
        if not violates(vdir, variant, n_inputs, include_dir):
            needed.append(g)
    info["present"] = present
    info["seconds"] = round(time.time() - t0, 1)
    return frozenset(needed), info


if __name__ == "__main__":
    import sys
    sig, info = signature(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50,
                          sys.argv[3] if len(sys.argv) > 3 else HERE)
    print(json.dumps({"signature": sorted(sig) if sig is not None else None, **info}))
