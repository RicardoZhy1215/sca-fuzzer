"""
Total time of the whole run behind the paper's Spectre v4 figure (10000 programs each), by stage:
  Revizor           measured wall time of its 4 runs (5595 s); confirmation = confirmed violations x
                    37.3 s (measured per violation in the timed Revizor run); the rest split by the timed
                    run's non-confirmation shares
  RL GRU / TF       the earlier A/B arms' measured time (tc.jsonl s, first 10000 programs), split by the
                    per-outcome shares of the from-scratch timed runs (estimate-old-ab.json)
  PPO updates       sum of upd_s in agent.jsonl over the first 10000 programs; they run in parallel on
                    other cores, so they are drawn separately and not added to the wall time

  python rvzr/SpecRL/genbandit/plot_time_total.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import time_breakdown as tb  # noqa: E402

R = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "figs", "v4-time-total")
MERGE = [("Generation + policy + inputs", ("Program generation", "Policy: waiting for agent", "Policy (RL)",
                                          "Input generation"), "#2a78d6"),
         ("Filters", ("Filters: hardware", "Filters: other"), "#1baf7a"),
         ("Fast path", ("Fast path: model", "Fast path: hardware", "Fast path: other"), "#6250d6"),
         ("Confirmation", ("Confirmation",), "#eda100"),
         ("Other (RL harness, logging, start-up)", ("Reporting", "RL harness", "Other"), "#b8b6ad")]
LEARN = "#eb6834"


def revizor():
    rows = [json.loads(l) for l in open(os.path.join(R, "ttk-default64-261107", "ttk.jsonl"))]
    wall = sum(r["wall_s"] for r in rows)
    viol = sum(r["violations"] for r in rows)
    x = os.path.join(R, "revizor-default64-extra247-261110-1555")
    import re
    txt = open(os.path.join(x, "fuzz.log"), errors="replace").read()
    wall += float(re.search(r"Duration: ([\d.]+)", txt).group(1))
    viol += len(glob.glob(os.path.join(x, "violation-*")))
    timed = json.load(open(os.path.join(R, "time-breakdown-v4-261112-1753", "revizor", "time_breakdown.json")))
    g, acc = tb.groups(timed)
    conf_timed = sum(v for (p, c), v in acc.items() if p.startswith("confirm"))
    per_violation = conf_timed / tb.nviol(timed)
    conf = viol * per_violation
    rest = {k: v for k, v in g.items() if k != "Confirmation"}
    s = sum(rest.values())
    out = {k: (wall - conf) * v / s for k, v in rest.items()}
    out["Confirmation"] = conf
    return out, wall, viol, per_violation


def rl():
    est = json.load(open(os.path.join(R, "time-scratch-v4-261112-1928", "estimate-old-ab.json")))
    res = {}
    for a in ("gru", "tf"):
        g = {k: v * 10000 / 1000 for k, v in est[a]["per_program_ms"].items()}      # seconds
        A = [json.loads(l) for l in open(os.path.join(R, "ctx-ab-gru-tf-default64-261107", a, "agent.jsonl"))]
        U = [x for x in A if x["results"] <= 10000]
        res[a] = (g, est[a]["wall_s"], sum(x["upd_s"] for x in U), len(U))
    return res


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rev, rev_wall, rev_v, per_v = revizor()
    rl_ = rl()
    rows = [("Revizor", rev, rev_wall, None), ("RL (GRU)", *rl_["gru"][:2], rl_["gru"][2:]),
            ("RL (Transformer)", *rl_["tf"][:2], rl_["tf"][2:])]
    lines = [f"Revizor: 10000 programs, {rev_v} confirmed violations x {per_v:.1f} s"]
    with open(OUT + ".csv", "w") as f:
        f.write("run,group,seconds,share\n")
        for name, g, wall, learn in rows:
            for lab, ks, _ in MERGE:
                v = sum(g.get(k, 0.0) for k in ks)
                f.write(f"{name},{lab},{v:.1f},{v / wall:.4f}\n")
                lines.append(f"{name:18s} {lab:40s} {v / 60:7.1f} min {100 * v / wall:5.1f}%")
            lines.append(f"{name:18s} {'TOTAL wall':40s} {wall / 60:7.1f} min ({wall / 3600:.2f} h)")
            if learn:
                f.write(f"{name},PPO updates (parallel),{learn[0]:.1f},{learn[0] / wall:.4f}\n")
                lines.append(f"{name:18s} {'PPO updates (parallel, other cores)':40s} {learn[0] / 60:7.1f} min "
                             f"({learn[1]} updates, {learn[0] / learn[1]:.2f} s each)")
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=200)
    fig.patch.set_facecolor("white")
    ys = [0, 1.25, 2.5]
    for (name, g, wall, learn), y in zip(rows, ys):
        left = 0.0
        for lab, ks, col in MERGE:
            v = sum(g.get(k, 0.0) for k in ks) / 60
            ax.barh(y, v, left=left, color=col, edgecolor="white", linewidth=1.2, height=0.55,
                    label=lab if y == 0 else None)
            left += v
        ax.text(left, y, f"  {wall / 3600:.2f} h", va="center", color=tb.TEXT, fontsize=9)
        if learn:
            ax.barh(y + 0.42, learn[0] / 60, color=LEARN, height=0.18, edgecolor="white", linewidth=0.8,
                    label="PPO updates (in parallel, other cores)" if y == ys[1] else None)
            ax.text(learn[0] / 60, y + 0.42, f"  {learn[0] / 60:.1f} min ({learn[1]} updates)", va="center",
                    color=tb.TEXT2, fontsize=7.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock time for 10,000 programs (minutes)", color=tb.TEXT)
    tb._style(ax)
    ax.set_xlim(0, max(r[2] for r in rows) / 60 * 1.15)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.22))
    fig.suptitle("Spectre v4: where the time of a 10,000-program run goes", x=0.02, ha="left", fontsize=11,
                 fontweight="bold", color=tb.TEXT)
    fig.tight_layout()
    fig.savefig(OUT + ".png", bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print(OUT + ".png")


if __name__ == "__main__":
    main()
