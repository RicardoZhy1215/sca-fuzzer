"""
Leak pattern of every Spectre v4 / v1 violation, extracted from the contract model (no test runs on the
executor: it is only asked for the sandbox base addresses).

For a violation's saved counterexample pair (the two inputs named in report.txt), the permissive contract
(CT-BPAS for v4, CT-COND for v1; nesting 1, then 5) is traced per input with a hooked Unicorn model that
logs every executed instruction (registers and flags before it), its memory accesses (with the values
read / written) and the stack of open speculations. The leak is the first contract-trace observation that
differs between the two inputs.
  source        the speculation that observation belongs to: the bypassed store (v4) / the mispredicted
                conditional branch (v1)
  transmitter   the instruction that made the observation: a load/store whose address differs (obs = mem),
                or the instruction before a control-flow difference (obs = branch)
  transmit part differential backward slice from the transmitter's address registers (or flags): only
                registers / flags / memory whose VALUES differ between the two inputs are followed, back
                to the start of the speculation. It ends at the secret: a load of memory whose content
                differs ("^", root = memory), the stale value itself ("*", root = stale), or registers
                that already differ when the speculation starts (root = register)
  access part   how the speculation steered the secret load (v4): forward taint from the bypassed stores'
                old values over the speculative path, then a backward slice of the secret load's inputs
                (address, other sources, flags) through tainted instructions; its roots are the loads of
                the stale value ("*"). steer = stale-pointer (found), stale-data (the stale value is the
                secret), no-stale (not found); v1: wrong-path (the secret load just runs on the wrong path)
Instrumentation (sandbox masks) is dropped from the abstracted chain. Abstraction: registers renamed in
order of appearance (R0, R1, ...; the sandbox base r14 is SB), immediates -> I, widths dropped,
cmovCC / setCC / jCC -> one name.
  L1 skeleton   speculation | steer | root | observation | transmitter access (load/store/rmw/branch)
  L2 opcodes    L1 + mnemonics:  source -> access part ~> transmit part => transmitter
  L3 full       the same with operands (registers renamed)

  cd ~/sca-fuzzer   (no campaign running)
  python rvzr/SpecRL/genbandit/leak_patterns.py v4 <out.tsv> [max per tool]   # or v1
"""
import os
import re
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)
from unicorn import UC_MEM_WRITE  # noqa: E402
from unicorn import x86_const as ucx  # noqa: E402
from rvzr.config import CONF  # noqa: E402
from rvzr import factory  # noqa: E402
from rvzr.arch.x86.target_desc import X86TargetDesc  # noqa: E402

NORM = X86TargetDesc.reg_normalized
GPR = {"A": "RAX", "B": "RBX", "C": "RCX", "D": "RDX", "SI": "RSI", "DI": "RDI", "RSP": "RSP", "RBP": "RBP",
       **{str(i): f"R{i}" for i in range(8, 16)}}
GPR = {k: getattr(ucx, "UC_X86_REG_" + v) for k, v in GPR.items() if k in set(NORM.values())}
FLAG_BIT = {"CF": 0, "PF": 2, "AF": 4, "ZF": 6, "SF": 7, "TF": 8, "IF": 9, "DF": 10, "OF": 11}
TOK = re.compile(r"[a-z][a-z0-9]*")
COLS = ["tool", "violation", "nesting", "frames", "obs", "source", "transmitter", "chain", "same_in_B",
        "L1", "L2", "L3"]


def addr_regs(expr):
    return {NORM[t] for t in TOK.findall(expr.lower()) if t in NORM}


def rw(insn):
    """registers / flags read and written (memory goes through the concrete accesses)"""
    R, W, RF, WF = set(), set(), set(), set()
    for op in list(insn.operands) + list(insn.implicit_operands):
        k = type(op).__name__
        if k == "RegisterOp":
            r = NORM.get(op.value)
            if r is None:
                continue
            if op.src:
                R.add(r)
            if op.dest:
                W.add(r)
                if getattr(op, "width", 64) in (8, 16):       # partial write keeps the old upper bits
                    R.add(r)
        elif k in ("MemoryOp", "AgenOp"):
            R |= addr_regs(op.value)
        elif k == "FlagsOp":
            RF |= set(op.get_flags_by_type("read"))
            WF |= set(op.get_flags_by_type("write"))
    if family(insn.name) == "cmovCC":                # the move may not happen: the old value flows through
        R |= W
    return R, W, RF, WF


def family(name):
    n = name.lower()
    for p, f in (("cmov", "cmovCC"), ("set", "setCC")):
        if n.startswith(p):
            return f
    if n.startswith("j") and n != "jmp":
        return "jCC"
    return n


def abstract(insn, canon):
    def reg(v):
        r = NORM.get(v)
        if r is None:
            return v
        if r == "14":
            return "SB"
        return canon.setdefault(r, f"R{len(canon)}")

    ops = []
    for op in insn.operands:
        k = type(op).__name__
        if k == "RegisterOp":
            ops.append(reg(op.value))
        elif k in ("MemoryOp", "AgenOp"):
            rs = [reg(t) for t in TOK.findall(op.value.lower()) if t in NORM]
            ops.append(("M[" if k == "MemoryOp" else "A[") + "+".join(rs) + "]")
        elif k == "ImmediateOp":
            ops.append("I")
        elif k == "LabelOp":
            ops.append("L")
    return family(insn.name) + (" " + ",".join(ops) if ops else "")


def access(insn, mem):
    w = any(x[0] for x in mem)
    r = any(not x[0] for x in mem)
    if insn is not None and insn.is_control_flow and not mem:
        return "branch"
    return "rmw" if (w and r) else "store" if w else "load" if r else "none"


class Extractor:
    def __init__(self, target, first_vdir):
        self.target = target
        CONF.load(os.path.join(first_vdir, "reproduce.yaml"), first_vdir)
        self.f = factory.get_fuzzer(os.path.join(ROOT, "base.json"), tempfile.mkdtemp(), "", None)
        saved = list(CONF.contract_execution_clause)
        CONF.contract_execution_clause = ["bpas" if target == "v4" else "cond"]
        self.m = factory.get_model(self.f.executor.read_base_addresses())
        CONF.contract_execution_clause = saved
        self.tmp = tempfile.mkdtemp()
        self._hook()

    # ------------------------------------------------------------------ hooks
    def _reset(self):
        self.log, self.meta, self.frames, self.stack = [], [], [], []
        self.pending_store = None

    def _hook(self):
        X = self
        m, sp, tr = self.m, self.m.speculator, self.m.tracer
        icb, mcb, cp, rb = m.instruction_callback, m.mem_access_callback, sp._checkpoint, sp.rollback

        def instruction_callback(self, address, size):
            em = self.emulator
            e = {"pc": address, "insn": None, "stack": tuple(X.stack), "mem": [],
                 "regs": {r: em.reg_read(c) for r, c in GPR.items()}, "fl": em.reg_read(ucx.UC_X86_REG_EFLAGS)}
            X.log.append(e)
            icb(address, size)
            if not self.state.is_exit_addr(address):
                e["insn"] = self.state.current_instruction

        def mem_access_callback(self, acc, address, size, value):
            if X.log:
                w = acc == UC_MEM_WRITE
                val = value if w else bytes(self.emulator.mem_read(address, size))
                X.log[-1]["mem"].append((w, address, size, val))
            mcb(acc, address, size, value)

        def checkpoint(self, *a, **k):
            fid = len(X.frames)
            ps = getattr(self, "_previous_store", None)
            if ps is not None:                       # BPAS: the store was the previous instruction; the
                src = len(X.log) - 2                 # current one is already on the speculative path
                X.frames.append({"kind": "bpas", "src": src, "mem": (ps[0], ps[1]), "start": len(X.log) - 1})
                X.log[-1]["stack"] = X.log[-1]["stack"] + (fid,)
            else:                                    # COND: the current instruction is the branch
                X.frames.append({"kind": "cond", "src": len(X.log) - 1, "mem": None, "start": len(X.log)})
            X.stack.append(fid)
            return cp(*a, **k)

        def rollback(self):
            if X.stack:
                X.stack.pop()
            return rb()

        def wrap(name, kind):
            orig = getattr(tr, name)

            def f(self, value):
                n = len(self.trace)
                orig(value)
                if len(self.trace) > n:
                    X.meta.append({"kind": kind, "seq": len(X.log) - 1, "stack": tuple(X.stack)})
            setattr(tr, name, types.MethodType(f, tr))

        m.instruction_callback = types.MethodType(instruction_callback, m)
        m.mem_access_callback = types.MethodType(mem_access_callback, m)
        sp._checkpoint = types.MethodType(checkpoint, sp)
        sp.rollback = types.MethodType(rollback, sp)
        wrap("_add_mem_address_to_trace", "mem")
        wrap("_add_pc_to_trace", "pc")

    def trace(self, inp, nest):
        self._reset()
        ct = self.m.trace_test_case([inp], nest)[0]
        typed = ct.get_typed()
        assert len(typed) == len(self.meta), (len(typed), len(self.meta))
        return typed, (self.log, self.meta, self.frames)

    # ------------------------------------------------------------------ one violation
    def check(self, vdir):
        ids = [int(x) for x in re.findall(r"^Input #(\d+)", open(os.path.join(vdir, "report.txt")).read(), re.M)][:2]
        pair = self.f.data_gen.load([os.path.join(vdir, f"input_{i:04}.bin") for i in ids])
        path = os.path.join(self.tmp, "p.asm")
        open(path, "w").write(open(os.path.join(vdir, "program.asm")).read())
        tc = self.f.asm_parser.parse_file(path, self.f.code_gen, self.f.elf_parser)
        self.m.load_test_case(tc)
        for nest in (1, 5):
            (ta, A), (tb, B) = self.trace(pair[0], nest), self.trace(pair[1], nest)
            i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), None)
            if i is None and len(ta) != len(tb):
                i = min(len(ta), len(tb))
            if i is not None:
                break
        if i is None:
            return {"obs": "no-difference"}
        _, A0 = self.trace(pair[0], 0)               # architectural execution of input A (for the access part)
        r = self.leak(A, B, i, len(ta) == i or len(tb) == i, A0[0])
        r["nesting"] = nest
        return r

    def leak(self, A, B, i, ended, arch):
        """differential backward slice: follow only registers / flags / memory whose values differ
        between the two inputs, from the transmitter back to the start of its speculation"""
        la, ma, frames = A
        lb = B[0]
        if ended:
            return {"obs": "length", "L1": "length", "L2": "length", "L3": "length"}
        ob = ma[i]
        t = ob["seq"]
        if ob["kind"] == "pc":                       # control flow differs: the previous instruction decided
            t -= 1
            while t >= 0 and la[t]["insn"] is None:
                t -= 1
        assert all(la[k]["pc"] == lb[k]["pc"] for k in range(t + 1)), "executions diverge before the leak"
        te, tins = la[t], la[t]["insn"]
        tstack = te["stack"]
        if not tstack:
            return {"obs": ob["kind"] + "-nonspec", "transmitter": f"{tins}", "L1": "nonspec", "L2": "nonspec",
                    "L3": "nonspec"}
        fr = [frames[f] for f in tstack]
        dreg = lambda k, rs: {r for r in rs if r in GPR and la[k]["regs"][r] != lb[k]["regs"][r]}  # noqa: E731
        dfl = lambda k, fs: {f for f in fs if ((la[k]["fl"] ^ lb[k]["fl"]) >> FLAG_BIT[f]) & 1}  # noqa: E731
        ov = lambda x, y: x[0] < y[0] + y[1] and y[0] < x[0] + x[1]  # noqa: E731
        active = lambda k: la[k]["insn"] is not None and tstack[:len(la[k]["stack"])] == la[k]["stack"]  # noqa: E731
        if ob["kind"] == "mem":
            regs = set()
            for op in list(tins.operands) + list(tins.implicit_operands):
                if type(op).__name__ == "MemoryOp":
                    regs |= addr_regs(op.value)
            need, needf = dreg(t, regs), set()
        else:
            R, _, RF, _ = rw(tins)
            need, needf = dreg(t, R), dfl(t, RF)
        need_mem, chain, mem_from = [], [], {}       # need_mem: (addr, size) whose content differs
        start = fr[0]["start"]
        for k in range(t - 1, start - 1, -1):
            if not active(k):
                continue
            R, W, RF, WF = rw(la[k]["insn"])
            mw = [(a, s_) for w, a, s_, _ in la[k]["mem"] if w]
            if not (W & need or WF & needf or any(ov(x, y) for x in mw for y in need_mem)):
                continue
            chain.append(k)
            need -= W
            needf -= WF
            need_mem = [y for y in need_mem if not any(ov(x, y) for x in mw)]
            need |= dreg(k, R)
            needf |= dfl(k, RF)
            for (w, a, s_, va), (_, _, _, vb) in zip(la[k]["mem"], lb[k]["mem"]):
                if not w and va != vb:               # this load read different memory content
                    need_mem.append((a, s_))
                    mem_from[(a, s_)] = k
        chain.reverse()
        byp = [f["mem"] for f in fr if f["kind"] == "bpas"]
        roots = {}
        for y in need_mem:                           # memory no store on the path wrote: the secret's origin
            roots[mem_from[y]] = "stale" if any(ov(y, b) for b in byp) else "memory"
        kinds_root = set(roots.values()) | ({"register"} if need or needf else set())
        root = "+".join(sorted(kinds_root)) or "none"

        # access part (BPAS): how the speculation steered the secret load(s). Compare the speculative
        # execution with the architectural one at the same instruction (straight-line code: one instance
        # per PC) and slice back from each secret load through values that differ spec vs arch: registers,
        # flags, memory written on the path, and stores the speculation sent to another address (the
        # architectural location then keeps its old content: "displaced", marked "!"). Roots: loads of a
        # bypassed store's location (the stale value, "*"). COND: the secret load simply runs on the
        # wrong path (access part empty).
        access_part, stale_loads, displaced = set(), set(), set()
        if byp:
            amap = {}
            for ai, e in enumerate(arch):
                if e["insn"] is not None and e["pc"] not in amap:
                    amap[e["pc"]] = ai
            src_arch = min((amap.get(la[f["src"]]["pc"], 0) for f in fr), default=0)

            def sdiff(k):
                """registers, flags, and memory reads (same address, other content) that differ spec vs arch"""
                ai = amap.get(la[k]["pc"])
                if ai is None:
                    return set(GPR), set(FLAG_BIT), []
                ae = arch[ai]
                regs = {r for r in GPR if la[k]["regs"][r] != ae["regs"][r]}
                fl = {f for f in FLAG_BIT if ((la[k]["fl"] ^ ae["fl"]) >> FLAG_BIT[f]) & 1}
                mem = [(x[1], x[2]) for x, y in zip(la[k]["mem"], ae["mem"])
                       if not x[0] and not y[0] and x[1] == y[1] and x[3] != y[3]]
                return regs, fl, mem

            def writer(pos, test):
                for j in range(pos - 1, start - 1, -1):
                    if active(j) and test(j):
                        return j
                return None

            work, seen = [], set()

            def add(j, regs_only=None):
                R, _, RF, _ = rw(la[j]["insn"])
                dr, df, dm = sdiff(j)
                if regs_only is not None:            # displaced store: only its address matters
                    R, RF, dm = regs_only, set(), []
                work.extend((j, "reg", r) for r in R & dr)
                work.extend((j, "fl", f) for f in RF & df)
                work.extend((j, "mem", m) for m in dm)

            for r_k, kind in roots.items():
                if kind != "memory":
                    continue
                add(r_k)
            while work:
                pos, kind, key = work.pop()
                if (pos, kind, key) in seen:
                    continue
                seen.add((pos, kind, key))
                if kind == "reg":
                    j = writer(pos, lambda j: key in rw(la[j]["insn"])[1])
                elif kind == "fl":
                    j = writer(pos, lambda j: key in rw(la[j]["insn"])[3])
                else:
                    # who wrote this location last: on the speculative path vs architecturally. Different
                    # instructions -> the architectural writer was sent elsewhere by the speculation
                    j = writer(pos, lambda j: any(w and ov((a_, s_), key) for w, a_, s_, _ in la[j]["mem"]))
                    hi = amap.get(la[pos]["pc"], len(arch))
                    aj = next((ai for ai in range(hi - 1, src_arch, -1) if arch[ai]["insn"] is not None and
                               any(w and ov((a_, s_), key) for w, a_, s_, _ in arch[ai]["mem"])), None)
                    if aj is not None and (j is None or la[j]["pc"] != arch[aj]["pc"]):
                        jd = writer(pos, lambda j: la[j]["pc"] == arch[aj]["pc"])
                        if jd is not None and jd not in access_part:
                            access_part.add(jd)
                            displaced.add(jd)
                            regs = set()
                            for op in list(la[jd]["insn"].operands) + list(la[jd]["insn"].implicit_operands):
                                if type(op).__name__ == "MemoryOp":
                                    regs |= addr_regs(op.value)
                            add(jd, regs_only=regs)
                        continue
                    if j is None:
                        if any(ov(key, b_) for b_ in byp):
                            stale_loads.add(pos)     # read the bypassed store's old value
                        continue
                if j is not None and j not in access_part:
                    access_part.add(j)
                    add(j)
        if byp:
            steer = ("stale-pointer" if stale_loads else ("stale-data" if "stale" in kinds_root else "no-stale"))
            if displaced and stale_loads:
                steer += "+displaced"
        else:
            steer = "wrong-path"
        access_part -= set(chain)
        keep_a = [k for k in sorted(access_part) if not la[k]["insn"].is_instrumentation]
        keep = [k for k in chain if not la[k]["insn"].is_instrumentation]
        canon = {}
        src = [f["src"] for f in fr]
        mark = lambda k: ("*" if (roots.get(k) == "stale" or k in stale_loads) else  # noqa: E731
                          "^" if roots.get(k) == "memory" else "!" if k in displaced else "")
        src_abs = [abstract(la[s_]["insn"], canon) for s_ in src]
        acc_abs = [mark(k) + abstract(la[k]["insn"], canon) for k in keep_a]
        ch_abs = [mark(k) + abstract(la[k]["insn"], canon) for k in keep]
        t_abs = abstract(tins, canon)
        kinds = "+".join(f["kind"] for f in fr)
        acc = access(tins, [x[:3] for x in te["mem"]]) if ob["kind"] == "mem" else "branch"
        L1 = f"{kinds}|{steer}|{root}|{ob['kind']}|{acc}"
        L2 = L1 + "|" + " ; ".join([family(la[s_]["insn"].name) for s_ in src] + ["->"] +
                                    [mark(k) + family(la[k]["insn"].name) for k in keep_a] + ["~>"] +
                                    [mark(k) + family(la[k]["insn"].name) for k in keep] + ["=>", family(tins.name)])
        L3 = f"{kinds}|{steer}|{root}|{ob['kind']}|" + " ; ".join(src_abs + ["->"] + acc_abs + ["~>"] + ch_abs +
                                                                   ["=>", t_abs])
        line = lambda k: f"{la[k]['insn']} @{la[k]['insn'].line_num() + 1}"  # noqa: E731
        return {"obs": ob["kind"], "frames": kinds, "source": " | ".join(line(s_) for s_ in src),
                "transmitter": line(t), "chain": " ; ".join(mark(k) + line(k) for k in sorted(access_part)) +
                " ~> " + " ; ".join(mark(k) + line(k) for k in chain),
                "same_in_B": "", "L1": L1, "L2": L2, "L3": L3}


def main():
    target, out = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    from violation_similarity import sets
    groups = sets()[target]
    ex = Extractor(target, next(iter(groups.values()))[0])
    done = set()
    if os.path.exists(out):
        done = {l.split("\t")[1] for l in open(out) if not l.startswith("tool\t")}
    new = not os.path.exists(out)
    with open(out, "a", buffering=1) as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        for tool, dirs in groups.items():
            for vdir in dirs[:limit]:
                key = os.path.relpath(vdir, HERE)
                if key in done:
                    continue
                try:
                    r = ex.check(vdir)
                except Exception as e:                                     # keep going; record it
                    r = {"obs": f"error: {type(e).__name__}: {e}"[:200].replace("\n", " ")}
                f.write("\t".join([tool, key] + [str(r.get(c, "")) for c in COLS[2:]]) + "\n")


if __name__ == "__main__":
    main()
