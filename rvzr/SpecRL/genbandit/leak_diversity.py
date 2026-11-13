"""
How different are the violations? Diversity of leak patterns (leak_patterns.py) per target and tool.

Pattern levels (coarse -> fine):
  L1 skeleton   speculation kind | how the speculation reaches the secret (stale pointer / stale data /
                wrong path) | where the secret comes from (memory / register / stale value) | observation
                (mem / branch) | transmitter access (load / store / rmw)
  core          L1 + instruction category (Revizor's: BINARY, LOGICAL, DATAXFER, CMOV, SETCC, SEMAPHORE,
                ...) of the key instructions only: source, stale loads (*), displaced stores (!), secret
                load (^), transmitter (intermediate arithmetic dropped)
  shape         L1 + number of instructions in the access part and in the transmit part
  L2 opcodes    L1 + the mnemonics of the chain
  L3 full       the abstracted chain (operands, registers renamed)
Metrics per set:
  distinct      number of distinct patterns
  rarefied      expected number of distinct patterns in a random subsample of n violations, n = the
                smallest set of the target (sets differ in size; more violations -> more patterns)
  subsamples    distinct patterns in 2000 random subsamples (without replacement) of 80% of the smallest
                set: 95% interval, and P(RL > Revizor) over pairs of draws. (Not a bootstrap with
                replacement: that favours the larger set even when every violation is unique.)
  Hill-1/2      effective number of patterns: exp(Shannon entropy) / inverse Simpson index
  only-here     patterns this set has and the other tool's sets do not
  distance      normalized edit distance between the L3 chains (token = one abstracted instruction) of
                every pair of violations, and to the nearest other violation of the set
Also: rarefaction curves (expected distinct patterns vs number of violations).

  python rvzr/SpecRL/genbandit/leak_diversity.py
"""
import collections
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "analysis", "leak-patterns")
FIG = os.path.join(HERE, "figs", "leak-diversity")
LEVELS = ["L1", "core", "shape", "L2", "L3"]
LABEL = {"L1": "skeleton", "core": "core", "shape": "shape", "L2": "opcode chain", "L3": "full chain"}
RNG = np.random.default_rng(0)
COL = {"Revizor": "#2a78d6", "RL (GRU)": "#eb6834", "RL (Transformer)": "#1baf7a"}


def _categories():
    import json
    cat = {}
    for e in json.load(open(os.path.join(HERE, "../../../base.json"))):
        cat.setdefault(e["name"].lower(), e["category"].split("-", 1)[1])
    return cat


CAT = _categories()


def category(tok):
    mark = tok[:1] if tok[:1] in "*^!" else ""
    name = tok[len(mark):].split()[-1]                    # drop a lock / rep prefix
    c = {"cmovCC": "CMOV", "setCC": "SETCC", "jCC": "COND_BR"}.get(name) or CAT.get(name, name)
    return mark + c


def load(target):
    rows, bad = [], collections.Counter()
    for l in open(os.path.join(D, f"{target}.tsv")):
        f = l.rstrip("\n").split("\t")
        if f[0] == "tool":
            cols = f
            continue
        r = dict(zip(cols, f))
        if not r.get("L3") or r["L1"] in ("length", "nonspec"):
            bad[r.get("obs", "")[:40] or r.get("L1")] += 1
            continue
        parts = r["L3"].split("|", 4)[4].split(" ; ")
        a, b = parts.index("->"), parts.index("~>")
        c = parts.index("=>")
        r["shape"] = r["L1"] + f"|acc{b - a - 1}|tx{c - b - 1}"
        ops = r["L2"].split("|", 5)[5].split(" ; ")
        x, z = ops.index("->"), ops.index("=>")
        r["core"] = r["L1"] + "|" + " ; ".join([category(o) for o in ops[:x]] + ["->"] +
                                               [category(o) for o in ops[x + 1:z] if o[:1] in "*^!"] +
                                               ["=>"] + [category(o) for o in ops[z + 1:]])
        r["tokens"] = parts
        rows.append(r)
    return rows, bad


def rarefy(counts, n):
    N = sum(counts)
    if n >= N:
        return float(len(counts))
    lc = lambda a, b: math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)  # noqa: E731
    tot = lc(N, n)
    return sum(1 - (math.exp(lc(N - c, n) - tot) if N - c >= n else 0.0) for c in counts)


def hill(counts):
    p = np.array(counts, float) / sum(counts)
    return float(np.exp(-(p * np.log(p)).sum())), float(1 / (p ** 2).sum())


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(len(a), len(b), 1)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out, curves, dists, boot = [], {}, {}, {}
    for target in ("v4", "v1"):
        rows, bad = load(target)
        tools = list(dict.fromkeys(r["tool"] for r in rows))
        by = {t: [r for r in rows if r["tool"] == t] for t in tools}
        n = min(len(v) for v in by.values())
        out.append(f"== Spectre {target}: " + ", ".join(f"{t} {len(by[t])}" for t in tools) +
                   (f"   (not extracted: {dict(bad)})" if bad else ""))
        out.append(f"   rarefied to n = {n} violations")
        for lev in LEVELS:
            out.append(f"  -- {LABEL[lev]}")
            sets = {t: collections.Counter(r[lev] for r in by[t]) for t in tools}
            for t in tools:
                c = sets[t]
                others = set().union(*[set(sets[o]) for o in tools if o != t and (o == "Revizor") != (t == "Revizor")])
                h1, h2 = hill(list(c.values()))
                only = sum(1 for p in c if p not in others)
                only_v = sum(v for p, v in c.items() if p not in others)
                out.append(f"     {t:17s} distinct {len(c):4d}  rarefied {rarefy(list(c.values()), n):6.1f}  "
                           f"Hill-1 {h1:6.1f}  Hill-2 {h2:6.1f}  singletons {sum(1 for v in c.values() if v == 1):4d}"
                           f"  only-here {only:4d} ({only_v} violations)")
                P = [r[lev] for r in by[t]]
                m = int(0.8 * n)                     # without replacement, below every set's size: with
                boot[(target, lev, t)] = np.array([len(set(RNG.choice(P, m, replace=False)))   # replacement
                                                   for _ in range(2000)])   # the larger set wins by construction
                xs = list(range(1, len(by[t]) + 1, max(1, len(by[t]) // 60)))
                curves[(target, lev, t)] = (xs, [rarefy(list(c.values()), x) for x in xs])
            bs = {t: boot[(target, lev, t)] for t in tools}
            out.append("     subsamples of %d without replacement (2000): " % int(0.8 * n) + "  ".join(
                f"{t} {np.mean(v):.1f} [{np.quantile(v, .025):.0f}-{np.quantile(v, .975):.0f}]" for t, v in bs.items()) +
                "   " + "  ".join(f"P({t} > Revizor) = {np.mean(bs[t][:, None] > bs['Revizor'][None, ::4]):.2f}"
                                  for t in tools if t != "Revizor"))
            if lev == "L1":
                allp = collections.Counter(r[lev] for r in rows)
                for p, _ in allp.most_common():
                    out.append("       " + "  ".join(f"{sets[t].get(p, 0):4d}" for t in tools) + f"   {p}")
        # distances on the full chains
        out.append("  -- edit distance between full chains (median [IQR]; nn = to the nearest other violation)")
        for t in tools:
            T = [r["tokens"] for r in by[t]]
            M = np.array([[edit(a, b) if i < j else 0.0 for j, b in enumerate(T)] for i, a in enumerate(T)])
            M = M + M.T
            iu = np.triu_indices(len(T), 1)
            np.fill_diagonal(M, np.inf)
            nn = M.min(1)
            dists[(target, t)] = (M[iu], nn)
            q = lambda a: f"{np.median(a):.3f} [{np.quantile(a, .25):.3f}-{np.quantile(a, .75):.3f}]"  # noqa: E731
            out.append(f"     {t:17s} pairwise {q(M[iu])}  nn {q(nn)}  identical-to-another {np.mean(nn == 0):.2f}")
        out.append("")
    txt = "\n".join(out)
    open(FIG + ".txt", "w").write(txt + "\n")
    print(txt)

    # rarefaction curves: rows = target, columns = level
    TEXT, TEXT2 = "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(2, len(LEVELS), figsize=(14, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    for r, target in enumerate(("v4", "v1")):
        for c, lev in enumerate(LEVELS):
            ax = axes[r, c]
            for t in COL:
                if (target, lev, t) in curves:
                    xs, ys = curves[(target, lev, t)]
                    ax.plot(xs, ys, color=COL[t], lw=2, label=t)
            ax.set_facecolor("white")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(TEXT2)
            ax.tick_params(colors=TEXT2, length=3, labelsize=7.5)
            if r == 0:
                ax.set_title(LABEL[lev], color=TEXT, fontsize=10, loc="left")
            if c == 0:
                ax.set_ylabel(f"Spectre {target}\ndistinct leak patterns", color=TEXT, fontsize=9)
            if r == 1:
                ax.set_xlabel("violations", color=TEXT, fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG + "-rarefaction.png")
    fig.savefig(FIG + "-rarefaction.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=200)
    fig.patch.set_facecolor("white")
    bins = np.linspace(0, 1, 41)
    for ax, target in zip(axes, ("v4", "v1")):
        for t in COL:
            if (target, t) in dists:
                ax.hist(dists[(target, t)][0], bins=bins, density=True, histtype="step", lw=1.8, color=COL[t], label=t)
        ax.set_facecolor("white")
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=TEXT2, length=3, labelsize=8)
        ax.set_title(f"Spectre {target}", color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("edit distance between leak chains (pairs of violations)", color=TEXT, fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG + "-distance.png")
    fig.savefig(FIG + "-distance.pdf")
    print(FIG + "-rarefaction.png", FIG + "-distance.png")


if __name__ == "__main__":
    sys.exit(main())
