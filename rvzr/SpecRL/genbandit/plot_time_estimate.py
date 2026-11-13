"""
Time breakdown per program for the three runs behind the paper's Spectre v4 figure: Revizor
(measured with time_breakdown.py) and the earlier GRU / Transformer A/B arms (first 10000 programs
each; estimated: each program's measured time, split by the per-outcome group shares of the
from-scratch timing runs -- see runs/time-scratch-v4-261112-1928/RESULTS.md).

  python rvzr/SpecRL/genbandit/plot_time_estimate.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import time_breakdown as tb  # noqa: E402

REV = os.path.join(HERE, "runs", "time-breakdown-v4-261112-1753", "revizor", "time_breakdown.json")
EST = os.path.join(HERE, "runs", "time-scratch-v4-261112-1928", "estimate-old-ab.json")
OUT = os.path.join(HERE, "figs", "v4-time-breakdown-ab")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = json.load(open(REV))
    g_rev, _ = tb.groups(r)
    rows = [("Revizor", {k: v / r["programs"] * 1000 for k, v in g_rev.items()})]
    est = json.load(open(EST))
    rows += [("RL (GRU)*", est["gru"]["per_program_ms"]), ("RL (Transformer)*", est["tf"]["per_program_ms"])]
    keys = list(g_rev.keys())
    with open(OUT + ".csv", "w") as f:
        f.write("component," + ",".join(n for n, _ in rows) + "\n")
        for k in keys:
            f.write(k + "," + ",".join(f"{d.get(k, 0.0):.3f}" for _, d in rows) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(6.4, 1.6 + 0.9 * len(rows)), dpi=200)
    fig.patch.set_facecolor("white")
    names = [n for n, _ in rows]
    left = [0.0] * len(rows)
    for k in keys:
        vals = [d.get(k, 0.0) for _, d in rows]
        if max(vals) <= 0:
            continue
        ax.barh(names, vals, left=left, color=tb.COL[k], edgecolor="white", linewidth=1.5, height=0.5, label=k)
        left = [a + b for a, b in zip(left, vals)]
    for i, (l, (_, d)) in enumerate(zip(left, rows)):
        conf = d.get("Confirmation", 0.0)
        ax.text(l, i, f"  {l:.0f} ms ({100 * conf / l:.0f}% confirmation)", va="center", color=tb.TEXT, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock time per program (ms)", color=tb.TEXT)
    tb._style(ax)
    ax.set_xlim(0, max(left) * 1.45)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.suptitle("Spectre v4: time breakdown per program", x=0.02, ha="left", fontsize=11, fontweight="bold",
                 color=tb.TEXT)
    fig.text(0.02, -0.02, "* estimated: measured time per program, split by the per-outcome shares of a "
             "separate timed run", fontsize=7.5, color=tb.TEXT2, ha="left")
    fig.tight_layout()
    fig.savefig(OUT + ".png", bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print(OUT + ".png")


if __name__ == "__main__":
    main()
