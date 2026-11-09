"""
Cumulative violations vs number of programs, one line per series. A series is one run directory or
several joined in order with '+' (a run cut short + the run that finished it). Revizor runs are
read from fuzz.log (V after every test case); ctxrl runs (tcenv) from steps.jsonl (new violations).

  python rvzr/SpecRL/genbandit/spectre_v1/plot_curve.py --out figs/v1yaml-3k \
      --series "Revizor=runs/revizor-v1yaml-3k-261109+runs/revizor-v1yaml-3k-part2-n1176-i50-261109-2156" \
      [--series "RL=runs/ctx-..."] [--title ...]
writes <out>.png, <out>.pdf and <out>.csv (programs, then one cumulative-violation column per series)
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4, fixed order
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def run_curve(run):
    """per-program cumulative violations of one run: list, index i = after program i + 1"""
    steps = os.path.join(run, "steps.jsonl")
    if os.path.exists(steps):
        out, v = [], 0
        for line in open(steps):
            v += bool(json.loads(line)["new_v"])
            out.append(v)
        return out
    text = open(os.path.join(run, "fuzz.log"), errors="replace").read().replace("\r", "\n")
    last = {}
    for n, s in re.findall(r"^(\d+)\s+\(.*Stats: (.*)$", text, re.M):
        last[int(n)] = int(re.search(r"\bV:(\d+)", s).group(1))
    out, v = [], 0
    for i in range(1, max(last) + 1):
        v = last.get(i, v)
        out.append(v)
    return out


def series_curve(spec):
    curve, joins = [], []
    for run in spec.split("+"):
        run = run if os.path.isabs(run) else os.path.join(HERE, run)
        c = run_curve(run)
        if curve:
            joins.append(len(curve))
        base = curve[-1] if curve else 0
        curve += [base + x for x in c]
    return curve, joins


ap = argparse.ArgumentParser()
ap.add_argument("--series", action="append", required=True, help="label=run[+run...]")
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="Spectre v1: cumulative violations vs programs")
ap.add_argument("--subtitle", default="")
args = ap.parse_args()
out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
os.makedirs(os.path.dirname(out), exist_ok=True)

S = []
for sp in args.series:
    label, runs = sp.split("=", 1)
    c, joins = series_curve(runs)
    S.append((label, c, joins))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
for i, (label, c, joins) in enumerate(S):
    col = SERIES_COLORS[i]
    xs = range(1, len(c) + 1)
    ax.plot(xs, c, color=col, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)
    ax.plot([len(c)], [c[-1]], "o", ms=6, color=col, mec="white", mew=2, zorder=4)
    # direct label at the line end: series name (if several) + final count, in text ink
    name = f"{label}: " if len(S) > 1 else ""
    ax.annotate(f"{name}{c[-1]} violations\nin {len(c):,} programs", (len(c), c[-1]),
                xytext=(-8, 10), textcoords="offset points", ha="right", va="bottom",
                color=TEXT, fontsize=9)
    for j in joins:
        ax.axvline(j, color=TEXT2, lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.annotate(f"run resumed after a\nmachine restart ({j:,})", (j, 0), xytext=(4, 4),
                    textcoords="offset points", color=TEXT2, fontsize=8, va="bottom")
ax.set_xlabel("Programs generated", color=TEXT)
ax.set_ylabel("Cumulative violations", color=TEXT)
ax.set_xlim(0, max(len(c) for _, c, _ in S) * 1.02)
ax.set_ylim(0, max(c[-1] for _, c, _ in S) * 1.15)
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=TEXT2, length=0)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.suptitle(args.title, x=0.02, ha="left", color=TEXT, fontsize=11, fontweight="bold")
if args.subtitle:
    ax.set_title(args.subtitle, loc="left", color=TEXT2, fontsize=9)
if len(S) > 1:
    ax.legend([lab for lab, _, _ in S], frameon=False, loc="upper left", labelcolor=TEXT)
fig.tight_layout()
fig.savefig(out + ".png")
fig.savefig(out + ".pdf")
n = max(len(c) for _, c, _ in S)
with open(out + ".csv", "w") as f:
    f.write("programs," + ",".join(lab for lab, _, _ in S) + "\n")
    for i in range(n):
        f.write(f"{i + 1}," + ",".join(str(c[i]) if i < len(c) else "" for _, c, _ in S) + "\n")
print(out + ".png")
