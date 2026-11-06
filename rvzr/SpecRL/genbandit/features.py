"""
Static features of a generated test case: the observation for per-test-case decisions.

Generic on purpose (structure, instruction mix, memory accesses, register dataflow), not tied to
one violation type; the network decides which ones matter. Computed from the TestCaseProgram
object before anything runs on the executor.
"""
import json
from typing import Dict, List

GPRS = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi"]
_SUBREGS = {"rax": ["eax", "ax", "al", "ah"], "rbx": ["ebx", "bx", "bl", "bh"],
            "rcx": ["ecx", "cx", "cl", "ch"], "rdx": ["edx", "dx", "dl", "dh"],
            "rsi": ["esi", "si", "sil"], "rdi": ["edi", "di", "dil"]}
FAMILY: Dict[str, str] = {r: r for r in GPRS}
FAMILY.update({s: r for r, subs in _SUBREGS.items() for s in subs})
SLOW = {"mul", "imul", "div", "idiv"}
NO_PAIR = 64  # store->load distance reported when the program has no such pair


def _index_regs(mem_op) -> List[str]:
    """GPR families used in a memory operand's address (the sandbox base r14 is not one)."""
    toks = mem_op.value.replace("-", "+").replace("*", "+").split("+")
    return [FAMILY[t.strip().lower()] for t in toks if t.strip().lower() in FAMILY]


class FeatureExtractor:
    def __init__(self, spec_path: str):
        self.categories = sorted({s["category"] for s in json.load(open(spec_path))})
        self.names = (["n_inst", "n_bb", "n_cond_br", "n_jmp"]
                      + [f"cat:{c}" for c in self.categories] + ["cat:other"]
                      + ["n_load", "n_store", "n_rmw", "n_lock", "n_slow"]
                      + [f"idx:{r}" for r in GPRS]
                      + ["n_idx_regs", "st_ld_pairs", "st_ld_min_dist", "ld_to_addr",
                         "slow_to_addr", "dep_depth"])
        self._cat_index = {c: 4 + i for i, c in enumerate(self.categories)}

    def __call__(self, test_case) -> List[float]:
        x = [0.0] * len(self.names)
        base = 4 + len(self.categories) + 1
        idx_base = base + 5
        pending_store: Dict[str, int] = {}    # family -> position of the last store indexed by it
        last_writer: Dict[str, str] = {}      # family -> "load" | "slow" | "other"
        depth: Dict[str, int] = {}
        pairs, min_dist, ld_addr, slow_addr, max_depth = 0, NO_PAIR, 0, 0, 0
        idx_used = set()
        pos = 0
        for bb in test_case.iter_basic_blocks():
            x[1] += 1
            for inst in bb:
                if inst.is_instrumentation or inst.category == "MACRO":  # measurement macros
                    continue
                name = inst.name.split()[-1]
                if inst.is_control_flow:
                    x[3 if name == "jmp" else 2] += 1
                    continue
                x[0] += 1
                x[self._cat_index.get(inst.category, base - 1)] += 1
                ops = inst.operands + inst.implicit_operands
                mems = [o for o in ops if type(o).__name__ == "MemoryOp"]
                is_load = any(o.src for o in mems)
                is_store = any(o.dest for o in mems)
                x[base] += is_load
                x[base + 1] += is_store
                x[base + 2] += is_load and is_store
                x[base + 3] += inst.name.startswith("lock")
                x[base + 4] += name in SLOW
                idx = [r for o in mems for r in _index_regs(o)]
                for r in idx:
                    x[idx_base + GPRS.index(r)] += 1
                    idx_used.add(r)
                    if last_writer.get(r) == "load":
                        ld_addr += 1
                    elif last_writer.get(r) == "slow":
                        slow_addr += 1
                # a read-modify-write loads before it stores
                if is_load:
                    for r in idx:
                        if r in pending_store:
                            pairs += 1
                            min_dist = min(min_dist, pos - pending_store[r])
                if is_store:
                    for r in idx:
                        pending_store[r] = pos
                srcs = {FAMILY[o.value.lower()] for o in ops
                        if type(o).__name__ == "RegisterOp" and o.src and o.value.lower() in FAMILY}
                dsts = {FAMILY[o.value.lower()] for o in ops
                        if type(o).__name__ == "RegisterOp" and o.dest and o.value.lower() in FAMILY}
                d = 1 + max([depth.get(r, 0) for r in srcs | set(idx)], default=0)
                writer = "load" if is_load else ("slow" if name in SLOW else "other")
                for r in dsts:
                    depth[r] = d
                    last_writer[r] = writer
                    pending_store.pop(r, None)
                max_depth = max(max_depth, d)
                pos += 1
        x[idx_base + 6:] = [len(idx_used), pairs, min_dist, ld_addr, slow_addr, max_depth]
        return x
