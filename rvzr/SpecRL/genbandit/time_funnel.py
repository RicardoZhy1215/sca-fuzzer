"""
Where do the programs go, and where does the time go? A per-program funnel of a Spectre v4 run,
from the time_breakdown.py runs (not averages: counts per exit point, total time per outcome, and the
distribution of the per-program time).

Revizor's pipeline per program (rvzr/fuzzer.py), each step can discard the program:
  speculation filter -> observation filter -> fast path (round with 10 reps; a candidate violation?)
  -> confirmation: nesting -> taint_mistake -> priming -> noise (re-measure at 50, 100, 500 reps)
     -> priming_large (priming at 500 reps) -> arch_mismatch -> violation reported
Sources:
  RL GRU / RL Transformer  runs/time-scratch-v4-261112-1928/{gru,tf}: 10000 programs each, trained from
                           scratch; programs.jsonl (exit point + time per group of every program),
                           steps.jsonl (new / duplicate violation), time_breakdown.json (time per
                           confirmation sub-stage)
  Revizor                  runs/time-breakdown-v4-261112-1753/revizor: 3000 programs; exit counts and
                           total time per stage only (that harness logged no per-program records for
                           Revizor)

  python rvzr/SpecRL/genbandit/time_funnel.py
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "runs")
SCR = os.path.join(R, "time-scratch-v4-261112-1928")
REV = os.path.join(R, "time-breakdown-v4-261112-1753", "revizor", "time_breakdown.json")
OUT = os.path.join(HERE, "figs", "v4-time-funnel")
EXITS = [("spec_filter", "discarded by the speculation filter"),
         ("observ_filter", "discarded by the observation filter"),
         ("fast_path", "discarded by the fast path (no candidate)"),
         ("fp_nesting", "candidate, rejected: nesting check"),
         ("fp_taint_mistakes", "candidate, rejected: taint check"),
         ("fp_priming", "candidate, rejected: priming"),
         ("fp_large_sample", "candidate, rejected: noise (larger samples)"),
         ("violation", "candidate, confirmed violation")]
SUB = ["nesting", "taint_mistake", "priming", "noise", "priming_large", "arch_mismatch"]
STAGE_PH = [("generate", "program generation"), ("inputs", "input generation"), ("policy", "policy (RL)"),
            ("filter", "filters"), ("fast", "fast path"), ("confirm", "confirmation"), ("report", "reporting"),
            ("rl-harness", "RL harness"), ("loop", "other")]


def phases(acc):
    ph = collections.defaultdict(float)
    for k, v in acc.items():
        ph[k.split("|")[0]] += v
    return ph


def q(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "-"
    return f"{np.median(x):.2f} s [{np.quantile(x, .1):.2f}-{np.quantile(x, .9):.2f}]"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = []
    arms = {}
    for arm, name in (("gru", "RL (GRU)"), ("tf", "RL (Transformer)")):
        P = [json.loads(l) for l in open(os.path.join(SCR, arm, "programs.jsonl"))]
        S = [json.loads(l) for l in open(os.path.join(SCR, arm, "steps.jsonl"))]
        assert [p["stage"] for p in P] == [s["stage"] for s in S]
        tb = json.load(open(os.path.join(SCR, arm, "time_breakdown.json")))
        arms[name] = {"P": P, "S": S, "acc": tb["acc"], "wall": tb["wall"], "n": len(P)}
    rv = json.load(open(REV))
    st = dict(rv["stats"])
    st["violation"] = st.pop("violations")
    arms_order = ["Revizor"] + list(arms)

    # ---------------- (1) funnel: exit point counts
    L.append("(1) Where each program leaves the pipeline (count, % of the arm's programs)")
    hdr = f"  {'':46s}" + "".join(f"{a:>24s}" for a in arms_order)
    L.append(hdr)
    counts = {"Revizor": {k: st.get(k, 0) for k, _ in EXITS}}
    for a in arms:
        c = collections.Counter(p["stage"] for p in arms[a]["P"])
        counts[a] = {k: c.get(k, 0) for k, _ in EXITS}
    N = {"Revizor": rv["programs"], **{a: arms[a]["n"] for a in arms}}

    def row(label, f):
        L.append(f"  {label:46s}" + "".join(f"{f(a):>24s}" for a in arms_order))

    row("programs", lambda a: f"{N[a]}")
    for k, lab in EXITS:
        row(lab, lambda a: f"{counts[a][k]} ({100 * counts[a][k] / N[a]:.1f}%)")
    cand = {a: sum(counts[a][k] for k, _ in EXITS[3:]) for a in arms_order}
    row("-> passed the filters (reached the fast path)", lambda a: f"{cand[a] + counts[a]['fast_path']}")
    row("-> entered confirmation (candidates)", lambda a: f"{cand[a]}")
    row("   of which confirmed", lambda a: f"{counts[a]['violation']} ({100 * counts[a]['violation'] / max(cand[a], 1):.0f}%)")
    row("   of which rejected", lambda a: f"{cand[a] - counts[a]['violation']} "
        f"({100 * (cand[a] - counts[a]['violation']) / max(cand[a], 1):.0f}%)")
    for a in arms:
        V = [s for s in arms[a]["S"] if s["stage"] == "violation"]
        L.append(f"  {a}: confirmed violations new {sum(s['new_v'] for s in V)}, duplicate "
                 f"{sum(s['dup_v'] for s in V)}; by mode " + ", ".join(f"{m} {c}" for m, c in
                                                                 collections.Counter(s["mode"] for s in V).items()))
    L.append("")

    # ---------------- (2) time per outcome (RL: exact per program; Revizor: per stage only)
    L.append("(2) Time spent on the programs of each outcome (total minutes, share of the run's wall time;"
             " per program: median [10th-90th percentile])")
    out_tot = {}
    for a in arms:
        P, W = arms[a]["P"], arms[a]["wall"]
        L.append(f"  {a} (wall {W / 60:.1f} min, {arms[a]['n']} programs)")
        by = collections.defaultdict(list)
        for p in P:
            by[p["stage"]].append(p["wall"])
        out_tot[a] = {}
        for k, lab in EXITS:
            x = by.get(k, [])
            if not x:
                continue
            out_tot[a][k] = sum(x)
            L.append(f"    {lab:46s} n={len(x):5d}  total {sum(x) / 60:7.1f} min ({100 * sum(x) / W:5.1f}%)"
                     f"   per program {q(x)}")
        conf_by = collections.defaultdict(list)
        for p in P:
            if p["stage"] not in ("spec_filter", "observ_filter", "fast_path"):
                conf_by[p["stage"]].append(p["g"].get("Confirmation", 0.0))
        L.append("    confirmation time alone, per candidate: " + "; ".join(
            f"{k}: {q(v)}" for k, v in conf_by.items()))
        rej = sum(sum(v) for k, v in conf_by.items() if k != "violation")
        ok = sum(conf_by.get("violation", []))
        L.append(f"    confirmation time on rejected candidates {rej / 60:.1f} min vs on confirmed violations "
                 f"{ok / 60:.1f} min ({100 * rej / (rej + ok):.1f}% wasted on false candidates)")
    L.append("")

    # ---------------- (3) stage view, all arms (exact totals from the timers)
    L.append("(3) Total time per stage of the pipeline (minutes, share of wall)")
    L.append(f"  {'':34s}" + "".join(f"{a:>24s}" for a in arms_order))
    PH = {"Revizor": (phases(rv["acc"]), rv["wall"])}
    for a in arms:
        PH[a] = (phases(arms[a]["acc"]), arms[a]["wall"])
    for key, lab in STAGE_PH:
        def v(a, key=key):
            ph, W = PH[a]
            s = sum(x for k, x in ph.items() if k == key or k.startswith(key + ":"))
            return f"{s / 60:.1f} ({100 * s / W:.1f}%)" if s > 0 else "-"
        L.append(f"  {lab:34s}" + "".join(f"{v(a):>24s}" for a in arms_order))
        if key == "confirm":
            for s_ in SUB:
                def vs(a, s_=s_):
                    ph, W = PH[a]
                    x = ph.get("confirm:" + s_, 0.0)
                    return f"{x / 60:.1f} ({100 * x / W:.1f}%)" if x > 0 else "-"
                L.append(f"    {s_:32s}" + "".join(f"{vs(a):>24s}" for a in arms_order))
    L.append(f"  {'wall':34s}" + "".join(f"{PH[a][1] / 60:>21.1f} min" for a in arms_order))
    L.append("")
    # who pays for each confirmation sub-stage: reached by
    L.append("(4) Candidates reaching each confirmation sub-stage, and its time per candidate that reached it")
    for a in arms_order:
        c = counts[a]
        reach = {"nesting": cand[a], "taint_mistake": cand[a] - c["fp_nesting"]}
        reach["priming"] = reach["taint_mistake"] - c["fp_taint_mistakes"]
        reach["noise"] = reach["priming"] - (c["fp_priming"] if a != "Revizor" else c["fp_priming"])
        reach["priming_large"] = reach["noise"] - c["fp_large_sample"]
        reach["arch_mismatch"] = c["violation"]
        ph = PH[a][0]
        L.append(f"  {a:17s} " + "  ".join(
            f"{s_} {reach[s_]} ({ph.get('confirm:' + s_, 0) / max(reach[s_], 1):.1f} s)" for s_ in SUB))
    L.append("  (Revizor's fp_priming lumps rejections at priming and at priming_large; here all are counted at"
             " priming. In the RL runs every fp_priming took 1.3-1.9 s, i.e. was rejected at priming.)")
    txt = "\n".join(L)
    open(OUT + ".txt", "w").write(txt + "\n")
    print(txt)

    # ---------------- figure
    TEXT, TEXT2 = "#0b0b0b", "#52514e"
    COLS = {"spec_filter": "#b8b6ad", "observ_filter": "#8f8d85", "fast_path": "#6250d6",
            "fp_taint_mistakes": "#eb6834", "fp_priming": "#f2a07a", "fp_large_sample": "#c4421a",
            "fp_nesting": "#ffd2bd", "violation": "#1baf7a"}
    SHORT = dict(spec_filter="spec. filter", observ_filter="obs. filter", fast_path="fast path",
                 fp_nesting="rejected: nesting", fp_taint_mistakes="rejected: taint", fp_priming="rejected: priming",
                 fp_large_sample="rejected: noise", violation="confirmed violation")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), dpi=200, gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    fig.patch.set_facecolor("white")
    # (a) counts per 1000 programs, stacked
    ax = axes[0]
    ys = np.arange(len(arms_order))
    for k, _ in EXITS:
        left = [sum(1000 * counts[a][kk] / N[a] for kk, _ in EXITS[:[e[0] for e in EXITS].index(k)]) for a in arms_order]
        vals = [1000 * counts[a][k] / N[a] for a in arms_order]
        ax.barh(ys, vals, left=left, color=COLS[k], edgecolor="white", linewidth=0.8, height=0.55, label=SHORT[k])
    for i, a in enumerate(arms_order):
        ax.text(1005, i, f"{cand[a]} candidates\n{counts[a]['violation']} confirmed", va="center", fontsize=7,
                color=TEXT2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{a}\n({N[a]} programs)" for a in arms_order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1250)
    ax.set_xlabel("programs (per 1000), by where they leave the pipeline", fontsize=8.5, color=TEXT)
    ax.set_title("(a) Where the programs go", loc="left", fontsize=10, color=TEXT)
    # (b) wall time per outcome (RL arms, exact)
    ax = axes[1]
    for i, a in enumerate(arms):
        left = 0.0
        for k, _ in EXITS:
            v = out_tot[a].get(k, 0.0) / 60
            ax.barh(i, v, left=left, color=COLS[k], edgecolor="white", linewidth=0.8, height=0.55)
            if v > 6:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white")
            left += v
        ax.text(left, i, f"  {left:.0f} min", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(list(arms), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("wall-clock minutes spent on the programs of each outcome", fontsize=8.5, color=TEXT)
    ax.set_title("(b) Where the time goes (RL runs, per program)", loc="left", fontsize=10, color=TEXT)
    # (c) confirmation time per candidate, by outcome
    ax = axes[2]
    rng = np.random.default_rng(0)
    ks = ["fp_taint_mistakes", "fp_priming", "fp_large_sample", "violation"]
    for j, a in enumerate(arms):
        for i, k in enumerate(ks):
            x = [p["g"].get("Confirmation", 0.0) for p in arms[a]["P"] if p["stage"] == k]
            y = i + (j - 0.5) * 0.3 + rng.uniform(-0.08, 0.08, len(x))
            ax.scatter(x, y, s=7, color=COLS[k], alpha=0.7, marker="o" if j == 0 else "^",
                       label=a if i == 3 else None, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([SHORT[k] for k in ks], fontsize=8)
    ax.set_xlabel("confirmation time per candidate (s, log)", fontsize=8.5, color=TEXT)
    ax.set_title("(c) Cost of one candidate", loc="left", fontsize=10, color=TEXT)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    for ax in axes:
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(TEXT2)
        ax.tick_params(colors=TEXT2, length=3, labelsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT + ".png")
    fig.savefig(OUT + ".pdf")
    print(OUT + ".png")


if __name__ == "__main__":
    main()
