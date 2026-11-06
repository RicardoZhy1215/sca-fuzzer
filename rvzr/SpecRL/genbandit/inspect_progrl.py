"""
What did a progrl policy learn? Samples programs from the trained policy (policy.pt) and from
the prior (= Revizor) and compares: instruction classes, the most up/down-weighted
instructions, address-register use, store->load pairs through the same address register,
loads whose destination register is later used as an address, and per-decision-kind KL.
No executor involved.

  python rvzr/SpecRL/genbandit/inspect_progrl.py rvzr/SpecRL/genbandit/runs/<run> [--n 2000]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import progrl  # noqa: E402


def program_facts(prog, table):
    nm, ns = table["n_nonmem"], table["n_store"]
    cls = ["N" if s < nm else ("S" if s < nm + ns else "L") for s, _, _ in prog]
    mem, dst = [], []
    for s, fams, _ in prog:
        m = d = None
        for sl, f in zip(table["slots"][s], fams):
            if sl["kind"] == progrl.K_MEM and m is None:
                m = f
            if sl["kind"] == progrl.K_REG and sl["dest"] and d is None:
                d = f
        mem.append(m)
        dst.append(d)
    out = {}
    for dist in (1, 2, 4, 8):          # store then load through the same address register
        out[f"st->ld same addr <= {dist}"] = sum(
            1 for i in range(len(prog)) if cls[i] == "S" and mem[i] is not None and any(
                cls[j] == "L" and mem[j] == mem[i] for j in range(i + 1, min(i + 1 + dist, len(prog)))))
    out["load dest later used as address"] = sum(
        1 for i in range(len(prog)) if cls[i] in "L" and dst[i] is not None and any(
            mem[j] == dst[i] for j in range(i + 1, min(i + 9, len(prog)))))
    return cls, mem, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    act = json.load(open(os.path.join(args.run, "header.json")))["action"]
    gen, table, specs, CONF = progrl._offline_generator(act)
    saved = json.load(open(os.path.join(args.run, "table.json")))
    assert saved["sig"] == table["sig"], "instruction set differs from the run's"
    res = {}
    for name in ("prior", "learned"):
        L = progrl.Learner(table, seed=7, frozen=True)
        if name == "learned":
            L.pol.load_state_dict(torch.load(os.path.join(args.run, "policy.pt")))
        progs, toks = [], []
        for i in range(0, args.n, 100):
            p, t = L.sampler.sample(min(100, args.n - i))
            progs += p
            toks += t
        spec = Counter(s for p in progs for s, _, _ in p)
        facts = Counter()
        clsc, memc = Counter(), Counter()
        for p in progs:
            c, m, o = program_facts(p, table)
            clsc.update(c)
            memc.update(x for x in m if x is not None)
            facts.update(o)
        with torch.no_grad():
            kls = []
            for i in range(0, len(toks), 200):
                _, kl, _, vt, _ = progrl.evaluate(L.pol, toks[i:i + 200])
                kind = torch.tensor([[t[0] for t in x] + [0] * (kl.shape[1] - len(x))
                                     for x in toks[i:i + 200]])
                kls.append((kl, vt, kind))
        res[name] = dict(spec=spec, cls=clsc, mem=memc, facts=facts, kls=kls, n=len(progs))
    P, Lr = res["prior"], res["learned"]
    tot = lambda c: sum(c.values())
    print(f"programs sampled per policy: {args.n}")
    print("instruction class share   prior   learned")
    for c, nm in (("N", "non-memory"), ("S", "store"), ("L", "load")):
        print(f"  {nm:12s}          {P['cls'][c] / tot(P['cls']):6.3f}  {Lr['cls'][c] / tot(Lr['cls']):6.3f}")
    print("address register share    " + "  ".join(f"{f:>5s}" for f in progrl.FAMS))
    for nm, r in (("prior", P), ("learned", Lr)):
        print(f"  {nm:8s}                " + "  ".join(f"{r['mem'][i] / tot(r['mem']):5.3f}" for i in range(6)))
    print("per program (mean)                     prior   learned   ratio")
    for k in P["facts"]:
        a, b = P["facts"][k] / P["n"], Lr["facts"][k] / Lr["n"]
        print(f"  {k:34s} {a:7.2f}  {b:7.2f}   x{b / max(a, 1e-9):.2f}")
    import torch
    for kd, nm in ((0, "instruction"), (1, "register"), (2, "address reg")):
        num = sum(float(kl[(kind == kd) & vt].sum()) for kl, vt, kind in Lr["kls"])
        den = sum(int(((kind == kd) & vt).sum()) for kl, vt, kind in Lr["kls"])
        print(f"KL(learned || prior) per {nm:11s} decision: {num / max(den, 1):.4f} nats ({den} decisions)")
    ts, tp = tot(Lr["spec"]), tot(P["spec"])
    ratio = {s: (Lr["spec"][s] / ts) / max(P["spec"][s] / tp, 0.5 / tp) for s in range(len(specs))}
    print("most up-weighted instructions (learned share / prior share; count in learned sample):")
    for s in sorted(ratio, key=lambda s: -ratio[s])[:12]:
        print(f"  x{ratio[s]:5.2f}  {table['names'][s]:40s} {Lr['spec'][s]}")
    print("most down-weighted:")
    for s in sorted(ratio, key=lambda s: ratio[s])[:8]:
        print(f"  x{ratio[s]:5.2f}  {table['names'][s]:40s} {Lr['spec'][s]}")


if __name__ == "__main__":
    main()
