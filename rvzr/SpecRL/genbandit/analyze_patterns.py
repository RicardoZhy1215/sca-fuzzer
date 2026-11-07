"""
What do the trained ctxrl policies write differently from Revizor, and why does it help?

Part 1 (offline, no executor): sample fresh programs from the prior (= Revizor's distribution)
and from each arm's final policy.pt, render them with Revizor's generator (install()), and compare
static features (features.py) and the instruction mix.
Part 2 (hardware outcomes already logged): over the fresh programs the run actually executed
(tc.jsonl features + stage), which features predict a violation (AUC, top vs bottom quartile).

  python rvzr/SpecRL/genbandit/analyze_patterns.py rvzr/SpecRL/genbandit/runs/<ab run> [--n 3000]
"""
import argparse
import json
import os
import sys
import warnings
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--n", type=int, default=3000)
args = ap.parse_args()
run = os.path.abspath(args.run)

import torch  # noqa: E402
torch.set_num_threads(2)
import ctxrl  # noqa: E402
import progrl  # noqa: E402
import space_v4 as space  # noqa: E402
from features import FeatureExtractor  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

header = json.load(open(os.path.join(run, "header.json")))
names = header["features"]
arms = sorted(d for d in os.listdir(run) if os.path.exists(os.path.join(run, d, "policy.pt")))

# ------------------------------------------------------------------ part 1: distribution shift
gen, table, specs, CONF = progrl._offline_generator(header["action"])
for a in arms:
    assert json.load(open(os.path.join(run, a, "table.json")))["sig"] == table["sig"], a
fx = FeatureExtractor(space.SPEC)


class Src:
    pass


src = Src()
src.table, src.specs, src.seed = table, specs, None
progrl.install(gen, src)
asm = os.path.join(run, ".patterns.asm")


def sample(arch, path, n, seed):
    h = json.load(open(os.path.join(run, arch, "agent_header.json"))) if arch else {}
    pol = ctxrl.make_policy(table, arch=h.get("arch", "gru"), **(h.get("tf") or {}))
    if path:
        pol.load_state_dict(torch.load(path))
    pol.eval()
    sm = ctxrl.CtxSampler(table, pol, np.random.default_rng(seed))
    c0 = ctxrl.ctx_vector(0, 8, None)
    progs, feats = [], []
    for i in range(0, n, 50):
        k = min(50, n - i)
        for j, (_, prog, _, _) in enumerate(sm.sample(np.stack([c0] * k), [None] * k)):
            for x in prog:
                x[3] = f"pat{seed}:{i + j}"
            src.cur = (0, prog)
            feats.append(fx(gen.create_test_case(asm)))
            progs.append(prog)
    return progs, np.array(feats)


S = {"revizor": sample(None, None, args.n, 1)}
for a in arms:
    S[a] = sample(a, os.path.join(run, a, "policy.pt"), args.n, 2)
if os.path.exists(asm):
    os.remove(asm)
cols = list(S)

KEY = ["n_load", "n_store", "n_rmw", "n_lock", "n_slow", "n_idx_regs", "st_ld_pairs",
       "st_ld_min_dist", "ld_to_addr", "slow_to_addr", "dep_depth"]
KEY += [f"idx:{r}" for r in progrl.FAMS]
KEY += [c for c in names if c.startswith("cat:")
        and any(S[k][1][:, names.index(c)].mean() >= 0.3 for k in cols)]
print(f"Part 1: {args.n} fresh programs per policy (feature means per program)")
print(f"  {'feature':24s}" + "".join(f"{c:>10s}" for c in cols) + "".join(f"{'x ' + c:>10s}" for c in cols[1:]))
for f in KEY:
    j = names.index(f)
    m = [S[c][1][:, j].mean() for c in cols]
    print(f"  {f:24s}" + "".join(f"{v:10.2f}" for v in m)
          + "".join(f"{v / max(m[0], 1e-9):10.2f}" for v in m[1:]))

nm, ns = table["n_nonmem"], table["n_store"]
for c in cols:
    cnt = Counter(s for p in S[c][0] for s, _, _, _ in p)
    tot = sum(cnt.values())
    S[c] = S[c] + (cnt, tot)
mn = lambda i: table["names"][i]
print("\n  instruction mix by mnemonic (share of all instructions)")
groups = Counter()
for i in range(len(table["names"])):
    groups[mn(i).split()[0] + (" (store)" if nm <= i < nm + ns else " (load)" if i >= nm + ns else "")] += 0
share = {c: Counter() for c in cols}
for c in cols:
    cnt, tot = S[c][2], S[c][3]
    for i, k in cnt.items():
        g = mn(i).split()[0] + (" (store)" if nm <= i < nm + ns else " (load)" if i >= nm + ns else "")
        share[c][g] += k / tot
rows = sorted(groups, key=lambda g: -max(share[c][g] for c in cols))
print(f"  {'mnemonic':22s}" + "".join(f"{c:>10s}" for c in cols))
for g in rows:
    if max(share[c][g] for c in cols) < 0.01:
        continue
    print(f"  {g:22s}" + "".join(f"{share[c][g]:10.3f}" for c in cols))

# ------------------------------------------------------------------ part 2: what predicts V
tc = [json.loads(l) for l in open(os.path.join(run, "tc.jsonl"))]
steps = {a: [json.loads(l) for l in open(os.path.join(run, a, "steps.jsonl"))] for a in arms}
X, y = [], []
for a in arms:
    rows_a = [r for r in tc if r.get("arm") == a]
    assert len(rows_a) >= len(steps[a]), (a, len(rows_a), len(steps[a]))
    for r, s in zip(rows_a, steps[a]):
        assert r["stage"] == s["stage"], "tc.jsonl and steps.jsonl out of order"
        if s["mode"] == "fresh":
            X.append(r["x"])
            y.append(r["stage"] == "violation")
X, y = np.array(X, float), np.array(y)
print(f"\nPart 2: hardware outcomes of {len(y)} fresh programs ({y.sum()} violations, "
      f"{1000 * y.mean():.1f}/1000), both arms, whole run")
print(f"  {'feature':24s} {'AUC':>6s}  V/1000 bottom quartile -> top quartile")
res = []
for f in KEY:
    j = names.index(f)
    v = X[:, j]
    if v.std() == 0:
        continue
    u = mannwhitneyu(v[y], v[~y]).statistic / (y.sum() * (~y).sum())
    lo, hi = np.quantile(v, [0.25, 0.75])
    bot, top = y[v <= lo].mean(), y[v >= hi].mean()
    res.append((abs(u - 0.5), f, u, bot, top, lo, hi))
for _, f, u, bot, top, lo, hi in sorted(res, reverse=True):
    print(f"  {f:24s} {u:6.3f}  {1000 * bot:5.1f} (<= {lo:g}) -> {1000 * top:5.1f} (>= {hi:g})")
