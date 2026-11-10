"""
Reward curve of ctxrl runs: new violations per 1000 programs in each update's batch (256
programs), as a rolling mean, against Revizor's rate on the same config; the policy's KL to its
prior underneath. One line per run (e.g. the arms of an A/B).

  python rvzr/SpecRL/genbandit/plot_reward.py --out figs/<name> --baseline 11.1 \
      --series "RL (GRU)=runs/<ab run>/gru" --series "RL (Transformer)=runs/<ab run>/tf"
writes <out>.png, <out>.pdf and <out>.csv
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.96):
    """95% Wilson interval of a binomial rate k / n, per 1000"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 1000 * max(c - h, 0.0), 1000 * (c + h)
SERIES_COLORS = ["#eb6834", "#1baf7a", "#2a78d6", "#eda100"]   # RL arms first; Revizor = slot-1 blue
REVIZOR = "#2a78d6"
TEXT, TEXT2 = "#0b0b0b", "#52514e"

ap = argparse.ArgumentParser()
ap.add_argument("--series", action="append", required=True, help="label=run dir (with agent.jsonl)")
ap.add_argument("--out", required=True)
ap.add_argument("--baseline", type=float, required=True, help="Revizor: violations per 1000 programs")
ap.add_argument("--baseline-label", default="Revizor")
ap.add_argument("--baseline-n", type=int, default=0,
                help="programs behind --baseline (draws its 95%% interval); 0 = no band")
ap.add_argument("--no-dots", action="store_true", help="don't plot the single updates")
ap.add_argument("--window", type=int, default=8, help="rolling mean over this many updates")
ap.add_argument("--max-programs", type=int, default=0)
ap.add_argument("--title", default="Spectre v4: reward during training")
args = ap.parse_args()
out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
os.makedirs(os.path.dirname(out), exist_ok=True)

S = []
for sp in args.series:
    label, run = sp.split("=", 1)
    run = run if os.path.isabs(run) else os.path.join(HERE, run)
    U = [json.loads(l) for l in open(os.path.join(run, "agent.jsonl"))]
    if args.max_programs:
        U = [u for u in U if u["results"] <= args.max_programs]
    x = [u["results"] for u in U]
    n = [sum(u["modes"].values()) for u in U]                 # programs in the update's batch
    k = [round(u["r_mean"] * m) for u, m in zip(U, n)]        # new violations in it
    r = [1000 * a / b for a, b in zip(k, n)]                  # new violations per 1000 programs
    w = args.window
    roll, lo, hi = [], [], []
    for i in range(len(r)):                                   # pooled over the last w updates
        K, N = sum(k[max(0, i - w + 1):i + 1]), sum(n[max(0, i - w + 1):i + 1])
        roll.append(1000 * K / N)
        a, b = wilson(K, N)
        lo.append(a)
        hi.append(b)
    S.append((label, x, r, roll, [u["kl"] for u in U], lo, hi))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, (ax, axk) = plt.subplots(2, 1, figsize=(6.4, 5.0), dpi=200, sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.12})
fig.patch.set_facecolor("white")
handles = []
for i, (label, x, r, roll, kl, lo, hi) in enumerate(S):
    col = SERIES_COLORS[i]
    if not args.no_dots:
        ax.plot(x, r, "o", ms=2.5, color=col, alpha=0.35, mew=0, zorder=2)
    ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0, zorder=1)
    h, = ax.plot(x, roll, color=col, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=3,
                 label=f"{label}, last {args.window} updates (95% CI)")
    handles.append(h)
    axk.plot(x, kl, color=col, lw=2, solid_joinstyle="round", solid_capstyle="round")
if args.baseline_n:
    blo, bhi = wilson(round(args.baseline * args.baseline_n / 1000), args.baseline_n)
    ax.axhspan(blo, bhi, color=REVIZOR, alpha=0.10, lw=0, zorder=0)
hb = ax.axhline(args.baseline, color=REVIZOR, lw=1.5, ls=(0, (4, 3)), zorder=1,
                label=f"{args.baseline_label}, same config ({args.baseline:g}"
                      + (f", 95% CI over {args.baseline_n:,} programs)" if args.baseline_n else ")"))
handles.append(hb)
ax.set_ylabel("New violations\nper 1000 programs", color=TEXT)
top = max(max(hi) for *_, hi in S) if args.no_dots else max(max(r) for _, _, r, *_ in S)
ax.set_ylim(0, top * 1.08)
axk.set_ylabel("KL to prior", color=TEXT)
axk.set_ylim(0, max(max(s[4]) for s in S) * 1.15)
axk.set_xlabel("Programs generated", color=TEXT)
for a in (ax, axk):
    a.set_facecolor("white")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(TEXT2)
        a.spines[s].set_linewidth(0.8)
    a.tick_params(colors=TEXT2, length=3, width=0.8)
axk.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
axk.set_xlim(0, max(x[-1] for _, x, *_ in S) * 1.02)
ax.legend(handles=handles, frameon=False, loc="upper left", labelcolor=TEXT, fontsize=8.5)
fig.suptitle(args.title, x=0.02, ha="left", color=TEXT, fontsize=11, fontweight="bold")
fig.subplots_adjust(left=0.13, right=0.97, top=0.9, bottom=0.1)
fig.savefig(out + ".png")
fig.savefig(out + ".pdf")
with open(out + ".csv", "w") as f:
    f.write("series,programs,reward_per_1000,rolling_mean,ci_low,ci_high,kl\n")
    for label, x, r, roll, kl, lo, hi in S:
        for row in zip(x, r, roll, lo, hi, kl):
            f.write(f"{label},{row[0]},{row[1]:.2f},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f},{row[5]:.5f}\n")
print(out + ".png")
