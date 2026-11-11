"""
Minimize SSE-FP violations on their own saved counterexample pair, then group the minimized
gadgets by shape: are many violations variants of one gadget?

For each violation directory (report.txt names the pair, input_XXXX.bin holds it):
  - the program body is split into units: an instruction plus the instrumentation lines in front
    of it (the `and reg, mask # instrumentation` that keeps its access in the sandbox)
  - units are removed greedily, last to first, repeated until nothing more can go. The stored pair
    (A, B) is contract-equivalent only for the ORIGINAL program, so every candidate gets its own
    pair, as Revizor's boosting builds one: A, and B' = B with the bytes the candidate's taint marks
    (under A) copied from A. A and B' keep every other difference of the original pair (where the
    secret is). A removal is kept if (A, B') is still a violation of the candidate: equal contract
    traces under the model AND different hardware traces in 2 of 2 trials
  - min.asm is checked once more (3 of 3) and written to <dir>/minimize/min.asm
The shape of a gadget = its instructions with registers renamed in order of first use and
immediates/displacements dropped; gadgets with the same shape are the same gadget.

  cd ~/sca-fuzzer   (executor free)
  python rvzr/SpecRL/genbandit/sse_fp/minimize.py <violation-dir>... [--reps 50] [--out shapes.tsv]
"""
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.getcwd())
from rvzr.config import CONF  # noqa: E402
from rvzr import factory  # noqa: E402

REG = re.compile(r"\b(r[a-d]x|r[sd]i|r[sd]p|r\d+[dwb]?|e[a-d]x|e[sd]i|[a-d][xlh]|[sd]il|[sd]i|xmm\d+)\b")
FAM = {}
for f, ns in {"rax": ["rax", "eax", "ax", "al", "ah"], "rbx": ["rbx", "ebx", "bx", "bl", "bh"],
              "rcx": ["rcx", "ecx", "cx", "cl", "ch"], "rdx": ["rdx", "edx", "dx", "dl", "dh"],
              "rsi": ["rsi", "esi", "si", "sil"], "rdi": ["rdi", "edi", "di", "dil"]}.items():
    for n in ns:
        FAM[n] = f


def units(lines):
    """indices of body lines grouped as [instrumentation..., instruction]"""
    out, inside, cur = [], False, []
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(".macro.measurement_start"):
            inside = True
            continue
        if s.startswith(".macro.measurement_end"):
            inside = False
            continue
        if not inside or not s or s.startswith((".", "#")):
            continue
        cur.append(i)
        if "instrumentation" not in s:
            out.append(cur)
            cur = []
    return out


def shape(lines):
    """normalized gadget: mnemonic + operands, registers renamed by first use, numbers dropped"""
    names, out = {}, []
    for line in lines:
        s = line.split("#")[0].strip()
        if not s or s.startswith(".") or "instrumentation" in line:
            continue

        def ren(m):
            r = m.group(1)
            k = "x" if r.startswith("xmm") else "g"
            key = r if k == "x" else FAM.get(r, r)
            if key in ("r14",):
                return "r14"
            names.setdefault(key, f"{k}{sum(1 for v in names.values() if v[0] == k)}")
            return names[key]
        s = REG.sub(ren, s)
        s = re.sub(r"-?\b(0x[0-9a-f]+|\d+)\b", "N", s)
        out.append(re.sub(r"\s+", " ", s))
    return " ; ".join(out)


class Checker:
    def __init__(self, vdir, reps):
        self.reps = reps
        CONF.load(os.path.join(vdir, "reproduce.yaml"), vdir)
        self.f = factory.get_fuzzer(os.path.join(os.getcwd(), "base.json"), tempfile.mkdtemp(), "", None)
        self.tmp = tempfile.mkdtemp()
        self.n = 0

    def pair(self, vdir):
        ids = [int(x) for x in re.findall(r"^Input #(\d+)", open(os.path.join(vdir, "report.txt")).read(),
                                          re.M)][:2]
        self.A, self.B = self.f.data_gen.load([os.path.join(vdir, f"input_{i:04}.bin") for i in ids])
        return ids

    def reboost(self):
        """B' = B with the bytes the loaded program's taint (under A) marks copied from A"""
        _, taints = self.f.model.trace_test_case_with_taints([self.A], CONF.model_max_nesting)
        b2 = self.B.copy()
        for actor in range(len(self.A)):
            t = taints[0].linear_view(actor)
            dst, src = b2.linear_view(actor), self.A.linear_view(actor)
            dst[t] = src[t]
        self.inputs = [self.A, b2]

    def violates(self, lines, trials):
        self.n += 1
        path = os.path.join(self.tmp, f"t{self.n % 4}.asm")
        open(path, "w").writelines(lines)
        try:
            tc = self.f.asm_parser.parse_file(path, self.f.code_gen, self.f.elf_parser)
            self.f.model.load_test_case(tc)
            self.reboost()
            ct = self.f.model.trace_test_case(self.inputs, CONF.model_max_nesting)
            if ct[0] != ct[1]:
                return False                     # the pair is no longer contract-equivalent
            self.f.executor.load_test_case(tc)
            for _ in range(trials):
                h = self.f.executor.trace_test_case(self.inputs, self.reps)
                if self.f.analyser.htraces_are_equivalent(h[0], h[1]):
                    return False
            return True
        except Exception:                        # unassemblable / model error: keep the unit
            return False


def minimize(chk, vdir):
    lines = open(os.path.join(vdir, "program.asm")).readlines()
    if not chk.violates(lines, 3):
        return None, lines, "orig-not-reproduced"
    changed = True
    while changed:                               # sweeps, last unit to first, until one removes nothing
        changed = False
        k = len(units(lines)) - 1
        while k >= 0:                            # removing unit k leaves units 0..k-1 where they are
            u = set(units(lines)[k])
            trial = [line for i, line in enumerate(lines) if i not in u]
            if chk.violates(trial, 2):
                lines, changed = trial, True
            k -= 1
    ok = chk.violates(lines, 3)
    return ok, lines, "ok" if ok else "min-not-stable"


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
        t0 = time.time()
        if chk is None:
            chk = Checker(vdir, reps)
        ids = chk.pair(vdir)
        ok, lines, status = minimize(chk, vdir)
        os.makedirs(os.path.join(vdir, "minimize"), exist_ok=True)
        open(os.path.join(vdir, "minimize", "min.asm"), "w").writelines(lines)
        body = [l for l in lines if l.strip() and not l.strip().startswith((".", "#")) and "instrumentation" not in l]
        row = (os.path.basename(os.path.dirname(vdir)), os.path.basename(vdir), f"#{ids[0]}/#{ids[1]}",
               status, str(len(body)), shape(lines), f"{time.time() - t0:.0f}s")
        print("\t".join(row), flush=True)
        rows.append(row)
        if out:
            with open(out, "w") as f:
                f.write("run\tviolation\tpair\tstatus\tinstructions\tshape\ttime\n")
                for r in rows:
                    f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    main()
