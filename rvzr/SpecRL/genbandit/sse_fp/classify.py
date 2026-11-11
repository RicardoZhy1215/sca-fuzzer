"""
Mechanism of a minimized SSE-FP violation: gated (205811 suppression), two-level FPVI, or direct VVI,
the same three classes as side-channel-fuzzer's 21-violation study (sse_fp/README section 9).

Input: <violation>/minimize/min.asm from minimize.py. The pair is rebuilt against min.asm as
minimize.py does (A, and B' = B with the bytes min.asm's taint marks copied from A).

  1. roles (gate_scan.py's test, hardware): per FP op, fence-i / neut-i / dead / gap-i ->
     PROBE (fence after it removes the difference) and GATE (needed, dead, outlives 512 nops)
  2. secret (hardware): delta debugging over the 8-byte words where A and B' differ, for a minimal
     set S such that A with S's words taken from B' is still a violation (equal contract traces
     under the model, different hardware traces 2 of 2). The secret = those words.
  3. where the secret goes (model + static data flow along A's architectural path): each secret
     word is a register's initial value (gpr:<reg>, simd:<xmm>), or a memory word that an
     architectural access touches (main+0x..., by which instruction) or that none does
     (transient-only). Then a forward taint from the secret through min.asm (register families,
     xmm registers, flags, and memory words using the model's per-instruction accesses) says which
     FP ops read secret-dependent values.
  4. class:
       two-level  every secret word is memory no architectural access touches, and no FP op reads it
       gated      a GATE op reads the secret and no PROBE op does
       direct     a PROBE op reads the secret
       other      none of the above (look at it)
Writes <violation>/minimize/classify.json and prints one tsv row.

  cd ~/sca-fuzzer   (executor free)
  python rvzr/SpecRL/genbandit/sse_fp/classify.py <violation-dir>... [--reps 50] [--out classes.tsv]
"""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimize import Checker  # noqa: E402
from gate_scan import neutralize, is_dead, differs  # noqa: E402
from triage import FP_OP  # noqa: E402
from rvzr.config import CONF  # noqa: E402
from rvzr.sandbox import SandboxLayout, DataArea  # noqa: E402

N_MAIN, N_FAULTY, N_GPR, N_SIMD = 512, 512, 8, 32
GPRS = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "flags", "rsp"]
FAM = {}
for f, ns in {"rax": "rax eax ax al ah", "rbx": "rbx ebx bx bl bh", "rcx": "rcx ecx cx cl ch",
              "rdx": "rdx edx dx dl dh", "rsi": "rsi esi si sil", "rdi": "rdi edi di dil",
              "rsp": "rsp esp sp spl", "rbp": "rbp ebp bp bpl"}.items():
    for n in ns.split():
        FAM[n] = f
MAIN0 = SandboxLayout.data_area_offset(DataArea.MAIN)
FAULTY0 = SandboxLayout.data_area_offset(DataArea.FAULTY)
MAX_TESTS = 150


def word_name(i):
    if i < N_MAIN:
        return f"main+0x{8 * i:03x}"
    if i < N_MAIN + N_FAULTY:
        return f"faulty+0x{8 * (i - N_MAIN):03x}"
    if i < N_MAIN + N_FAULTY + N_GPR:
        return f"gpr:{GPRS[i - N_MAIN - N_FAULTY]}"
    j = i - N_MAIN - N_FAULTY - N_GPR
    return f"simd:xmm{j // 4}.{'lo' if j % 4 == 0 else 'hi' if j % 4 == 1 else 'ymm'}"


def word_of_offset(off):
    """sandbox data offset -> input word index (main/faulty only)"""
    if MAIN0 <= off < MAIN0 + 8 * N_MAIN:
        return (off - MAIN0) // 8
    if FAULTY0 <= off < FAULTY0 + 8 * N_FAULTY:
        return N_MAIN + (off - FAULTY0) // 8
    return None


def ddmin(items, test, budget):
    """1-minimal subset of items for which test() holds (test(items) is assumed true)"""
    n, calls = 2, 0
    while len(items) >= 2 and calls < budget:
        size = math.ceil(len(items) / n)
        parts = [items[i:i + size] for i in range(0, len(items), size)]
        done = False
        for p in parts:
            calls += 1
            if test(p):
                items, n, done = p, 2, True
                break
        if not done and n > 2:
            for p in parts:
                calls += 1
                comp = [x for x in items if x not in set(p)]
                if test(comp):
                    items, n, done = comp, max(n - 1, 2), True
                    break
        if not done:
            if n >= len(items):
                break
            n = min(len(items), 2 * n)
    return items, calls


def body_instructions(tc):
    """measured instructions in program order (no macros)"""
    out = []
    for bb in tc.iter_basic_blocks():
        for inst in list(bb) + list(bb.terminators):
            if inst.name.startswith("macro") or inst.name == "jmp":
                continue
            out.append(inst)
    return out


def asm_instruction_lines(lines):
    idx, inside = [], False
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith(".macro.measurement_start"):
            inside = True
            continue
        if s.startswith(".macro.measurement_end"):
            inside = False
            continue
        if inside and s and not s.startswith((".", "#")) and not s.startswith("jmp"):
            idx.append(i)
    return idx


def reg_key(name):
    name = name.lower()
    return name if name.startswith("xmm") else FAM.get(name)


def flow(insts, accesses, secret_words):
    """forward taint from the secret along the architectural path; returns per-instruction
    'reads secret' flags"""
    regs = set()
    mem = set()
    for w in secret_words:
        if w < N_MAIN + N_FAULTY:
            mem.add(w)
        elif w < N_MAIN + N_FAULTY + N_GPR:
            regs.add(GPRS[w - N_MAIN - N_FAULTY])
        else:
            regs.add(f"xmm{(w - N_MAIN - N_FAULTY - N_GPR) // 4}")
    reads = []
    for inst, acc in zip(insts, accesses):
        ops = list(inst.operands) + list(inst.implicit_operands)
        tin = False
        for o in ops:
            t = type(o).__name__
            if t == "RegisterOp" and o.src:
                tin |= reg_key(o.value) in regs
            elif t == "MemoryOp" and o.src:
                tin |= any(w in mem for w in acc)
            elif t == "AgenOp":
                tin |= any(reg_key(r) in regs for r in re.findall(r"[a-z]+\d*", o.value))
            elif t == "FlagsOp":
                if any("r" in v for v in o._flag_values):
                    tin |= "flags" in regs
        reads.append(tin)
        for o in ops:
            t = type(o).__name__
            if t == "RegisterOp" and o.dest:
                k = reg_key(o.value)
                if k is None:
                    continue
                partial = (k.startswith("xmm") and re.search(r"(ss|sd)$", inst.name)) or \
                          (not k.startswith("xmm") and o.width < 32)
                if tin:
                    regs.add(k)
                elif not partial and not o.src:
                    regs.discard(k)
            elif t == "MemoryOp" and o.dest:
                for w in acc:
                    if tin:
                        mem.add(w)
            elif t == "FlagsOp" and any("w" in v for v in o._flag_values):
                if tin:
                    regs.add("flags")
                else:
                    regs.discard("flags")
    return reads


def classify(chk, vdir):
    lines = open(os.path.join(vdir, "minimize", "min.asm")).readlines()
    path = os.path.join(vdir, "minimize", "min.asm")
    tc = chk.f.asm_parser.parse_file(path, chk.f.code_gen, chk.f.elf_parser)
    chk.f.model.load_test_case(tc)
    chk.reboost()
    A, Bp = chk.inputs
    wd = os.path.join(vdir, "classify")
    os.makedirs(wd, exist_ok=True)
    d0 = differs(chk, lines, f"{wd}/orig.asm")
    res = {"violation": os.path.basename(vdir), "min_differs": d0}
    if not d0:
        res["class"] = "min-not-reproduced"
        return res

    # 1. roles
    fp_lines = [i for i, l in enumerate(lines) if FP_OP.match(l.strip())]
    gates, probes = [], []
    for i in fp_lines:
        df = differs(chk, lines[:i + 1] + ["lfence\n"] + lines[i + 1:], f"{wd}/fence-{i}.asm")
        repl = neutralize(lines[i])
        dn = differs(chk, lines[:i] + ([repl] if repl else []) + lines[i + 1:], f"{wd}/neut-{i}.asm")
        dg = None
        if is_dead(lines, i) and not dn:
            dg = differs(chk, lines[:i + 1] + ["nop\n"] * 512 + lines[i + 1:], f"{wd}/gap-{i}.asm")
        if dg:
            gates.append(i)
        elif not df:
            probes.append(i)
    chk.f.executor.load_test_case(tc)                    # back to min.asm for the secret search

    # 2. secret
    la, lb = A.linear_view(0), Bp.linear_view(0)
    n_words = N_MAIN + N_FAULTY + N_GPR + N_SIMD
    diff = [i for i in range(n_words) if la[i] != lb[i]]

    def test(subset):
        x = A.copy()
        lx = x.linear_view(0)
        lx[subset] = lb[subset]
        ct = chk.f.model.trace_test_case([A, x], CONF.model_max_nesting)
        if ct[0] != ct[1]:
            return False
        for _ in range(2):
            h = chk.f.executor.trace_test_case([A, x], chk.reps)
            if chk.f.analyser.htraces_are_equivalent(h[0], h[1]):
                return False
        return True
    secret, calls = ddmin(diff, test, MAX_TESTS)
    res["n_diff_words"], res["ddmin_tests"] = len(diff), calls
    res["secret"] = [word_name(w) for w in secret]
    res["secret_confirmed"] = test(secret)

    # 3. where it goes: per-instruction accesses of A (model), mapped to asm lines
    ct = chk.f.model.trace_test_case([A], CONF.model_max_nesting)[0]
    insts = body_instructions(tc)
    by_pc = {inst.section_offset(): k for k, inst in enumerate(insts)}
    accesses, cur = [set() for _ in insts], None
    for e in ct._trace:
        if e.type_ == "pc":
            cur = by_pc.get(e.value)
        elif e.type_ == "mem" and cur is not None:
            for off in (e.value, e.value + 7):
                w = word_of_offset(off)
                if w is not None:
                    accesses[cur].add(w)
    asm_idx = asm_instruction_lines(lines)
    ok_map = len(asm_idx) == len(insts) and all(
        lines[a].split()[0].lower() == insts[k].name.split()[-1] or lines[a].split()[0].lower() in insts[k].name
        for k, a in enumerate(asm_idx))
    res["map_ok"] = ok_map
    line_of = dict(enumerate(asm_idx))
    touched = {}
    for k, acc in enumerate(accesses):
        for w in acc:
            touched.setdefault(w, []).append(lines[line_of[k]].split("#")[0].strip() if k in line_of else "?")
    res["secret_where"] = []
    for w in secret:
        if w < N_MAIN + N_FAULTY:
            res["secret_where"].append(f"{word_name(w)} " + (f"read by: {' | '.join(touched[w])}" if w in touched
                                                             else "TRANSIENT-ONLY"))
        else:
            res["secret_where"].append(word_name(w))
    reads = flow(insts, accesses, secret)
    reads_line = {line_of[k]: r for k, r in enumerate(reads) if k in line_of}

    def ops(idx):
        return [lines[i].split("#")[0].strip() for i in idx]
    res["gates"], res["probes"] = ops(gates), ops(probes)
    res["fp_reading_secret"] = ops([i for i in fp_lines if reads_line.get(i)])
    gate_tainted = any(reads_line.get(i) for i in gates)
    probe_tainted = any(reads_line.get(i) for i in probes)
    mem_only = all(w < N_MAIN + N_FAULTY for w in secret)
    transient_only = mem_only and secret and all(w not in touched for w in secret)
    if not res["secret_confirmed"] or not secret:
        cls = "other(secret not isolated)"
    elif transient_only and not res["fp_reading_secret"]:
        cls = "two-level"
    elif gate_tainted and not probe_tainted:
        cls = "gated"
    elif probe_tainted:
        cls = "direct"
    else:
        cls = "other"
    res["class"] = cls
    return res


def main():
    args, reps, out = sys.argv[1:], 50, None
    for flag in ("--reps", "--out"):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1]
            del args[i:i + 2]
            reps, out = (int(val), out) if flag == "--reps" else (reps, val)
    chk = None
    for vdir in map(os.path.abspath, args):
        if chk is None:
            chk = Checker(vdir, reps)
        chk.pair(vdir)
        try:
            r = classify(chk, vdir)
        except Exception as e:                          # keep going; record the failure
            r = {"violation": os.path.basename(vdir), "class": f"error: {type(e).__name__}: {e}"[:200]}
        r["run"] = os.path.basename(os.path.dirname(vdir))
        json.dump(r, open(os.path.join(vdir, "minimize", "classify.json"), "w"), indent=1)
        row = (r["run"], r["violation"], r["class"], " ; ".join(r.get("secret_where", [])),
               " | ".join(r.get("gates", [])), " | ".join(r.get("probes", [])),
               " | ".join(r.get("fp_reading_secret", [])))
        print("\t".join(row), flush=True)
        if out:
            new = not os.path.exists(out)
            with open(out, "a") as f:
                if new:
                    f.write("run\tviolation\tclass\tsecret\tgates\tprobes\tFP ops reading the secret\n")
                f.write("\t".join(row) + "\n")


if __name__ == "__main__":
    main()
