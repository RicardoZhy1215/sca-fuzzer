"""
Is a minimized SSE-FP violation the suppression (205811 / "gated") shape: a dead FP op whose
secret-dependent underflow suppresses a later FPVI probe, with an effect that outlives the ROB?

Port of side-channel-fuzzer sse_fp/gate_scan.py, run on minimize.py's output (minimize/min.asm)
with the pair rebuilt against min.asm the same way minimize.py builds it (A, and B with the bytes
min.asm's taint marks copied from A). For every FP op i, on the hardware:
  fence-i   lfence right after op i                      -> difference gone: op i is the probe
  neut-i    op i replaced by a plain move of the same operands (movups/movss/movsd/movaps), so its
            own load stays and only the FP operation goes; dropped if it has no xmm destination
                                                          -> difference gone: op i's operation matters
  dead      op i's destination is not read by any later instruction (static)
  gap-i     512 nops right after op i                    -> difference kept: op i's effect lasts
GATE: neut-i removes the difference, op i is dead, and gap-i keeps it.

  cd ~/sca-fuzzer   (executor free)
  python rvzr/SpecRL/genbandit/sse_fp/gate_scan.py <violation-dir>... [--reps 50] [--out gates.tsv]
"""
import os
import re
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimize import Checker  # noqa: E402
from triage import FP_OP  # noqa: E402

GPR_FAMILY = {r: fam for fam, names in {
    "rax": "rax eax ax al ah", "rbx": "rbx ebx bx bl bh", "rcx": "rcx ecx cx cl ch",
    "rdx": "rdx edx dx dl dh", "rsi": "rsi esi si sil", "rdi": "rdi edi di dil"}.items()
    for r in names.split()}


def neutralize(line):
    body = line.split("#")[0].strip()
    m = re.match(r"(\S+)\s+(xmm\d+)\s*,\s*([^,]+?)(\s*,.*)?$", body)
    if not m:
        return None
    dst, src = m.group(2), m.group(3).strip()
    if "ptr" in src:
        mov = {"xmmword": "movups", "qword": "movsd", "dword": "movss"}.get(src.split()[0])
        return f"{mov} {dst}, {src}\n" if mov else None
    return f"movaps {dst}, {src}\n"


def is_dead(lines, i):
    body = lines[i].split("#")[0].strip()
    m = re.match(r"\S+\s+([a-z0-9]+)", body)
    if not m:
        return False
    reg = m.group(1)
    names = [n for n, f in GPR_FAMILY.items() if f == GPR_FAMILY.get(reg)] or [reg]
    later = [l.split("#")[0] for l in lines[i + 1:]]
    return not any(re.search(rf"\b{n}\b", l) for l in later for n in names)


def differs(chk, lines, path):
    open(path, "w").writelines(lines)
    tc = chk.f.asm_parser.parse_file(path, chk.f.code_gen, chk.f.elf_parser)
    chk.f.executor.load_test_case(tc)
    h = chk.f.executor.trace_test_case(chk.inputs, chk.reps)
    return not chk.f.analyser.htraces_are_equivalent(h[0], h[1])


def main():
    args, reps, out = sys.argv[1:], 50, None
    for flag in ("--reps", "--out"):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1]
            del args[i:i + 2]
            reps, out = (int(val), out) if flag == "--reps" else (reps, val)
    chk, rows = None, []
    for vdir in map(os.path.abspath, args):
        if chk is None:
            chk = Checker(vdir, reps)
        chk.pair(vdir)
        lines = open(os.path.join(vdir, "minimize", "min.asm")).readlines()
        path = os.path.join(vdir, "minimize", "min.asm")
        tc = chk.f.asm_parser.parse_file(path, chk.f.code_gen, chk.f.elf_parser)
        chk.f.model.load_test_case(tc)
        chk.reboost()                                   # the pair for min.asm
        wd = os.path.join(vdir, "gate_scan")
        os.makedirs(wd, exist_ok=True)
        d0 = differs(chk, lines, f"{wd}/orig.asm")
        gates, probes = [], []
        for i, l in enumerate(lines):
            if not FP_OP.match(l.strip()):
                continue
            op = l.split("#")[0].strip()
            df = differs(chk, lines[:i + 1] + ["lfence\n"] + lines[i + 1:], f"{wd}/fence-{i}.asm")
            repl = neutralize(l)
            dn = differs(chk, lines[:i] + ([repl] if repl else []) + lines[i + 1:], f"{wd}/neut-{i}.asm")
            dead = is_dead(lines, i)
            dg = None
            if d0 and dead and not dn:
                dg = differs(chk, lines[:i + 1] + ["nop\n"] * 512 + lines[i + 1:], f"{wd}/gap-{i}.asm")
            if dg:
                gates.append(op)
            elif d0 and not df:
                probes.append(op)
        row = (os.path.basename(os.path.dirname(vdir)), os.path.basename(vdir),
               "differs" if d0 else "SAME", "GATE" if gates else "-", " | ".join(gates), " | ".join(probes))
        print("\t".join(row), flush=True)
        rows.append(row)
        if out:
            with open(out, "w") as f:
                f.write("run\tviolation\tmin.asm on rebuilt pair\tgated\tgate ops\tprobe ops\n")
                for r in rows:
                    f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    main()
