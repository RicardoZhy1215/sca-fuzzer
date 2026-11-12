"""
Time breakdown of a fuzzing run: where the wall-clock time per program goes, for unmodified
Revizor and for ctxrl (tcenv --policy ctx), measured the same way on both.

Timers are attached at run time (monkeypatching instances / classes); no Revizor source is
changed. Every timer records EXCLUSIVE time (its own time minus that of nested timers) under
(phase, component):
  phase      generate (program generation incl. assembling), inputs (input generation),
             filter (speculation + observation filter), fast (fast path of the fuzzing round),
             confirm:<stage> (nesting / taint_mistake / priming / noise / priming_large /
             arch_mismatch), report (storing / printing a violation),
             policy (RL: before_generate = waiting for / installing the policy's program,
             observe = sending the outcome back), rl-harness (tcenv's feature extraction and
             hardware-observation summary), loop (anything else in the loop)
  component  model (Unicorn contract tracing incl. taint), boost (input boosting),
             hardware (executor load + trace), assemble (fenced copy for the observation
             filter), analysis (trace comparison), self (the phase's own remaining code)
Loop time not covered by any timer is reported as loop/self (logging, bookkeeping).

  cd ~/sca-fuzzer
  taskset -c 1-3 python rvzr/SpecRL/genbandit/time_breakdown.py revizor <out> [N]
  taskset -c 1-3 python rvzr/SpecRL/genbandit/time_breakdown.py rl <out> [N] [--init policy.pt|none] [--arch gru|tf]
  python rvzr/SpecRL/genbandit/time_breakdown.py report <name=dir>... --fig <path prefix>
  python rvzr/SpecRL/genbandit/time_breakdown.py blocks <name=dir>... --fig <path prefix> [--block 1000]
  taskset -c 4-7 python rvzr/SpecRL/genbandit/time_breakdown.py latency <name=dir>...
RL runs also write programs.jsonl: per program, its stage and the time of each group (below), so
the breakdown can be followed over training. A blocking wait of the fuzzer for the agent's next
program (prog_q.get with nothing ready) is timed separately as policy/wait.
"""
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
CFG = os.path.join(HERE, "v4", "c441d47e7c.yaml")
ACTION = {"program_size": 64, "mem_ratio": 0.5, "n_regs": 6, "entropy_bits": 10, "n_inputs": 50,
          "value_mix": "default", "binary": "on", "flags": "on", "semaphore": "on", "locks": "on",
          "spec_filter": "on"}
WARM = os.path.join(HERE, "runs", "ctx-ab-gru-tf-default64-261107", "gru", "policy.pt")


class T:
    acc = collections.defaultdict(float)        # (phase, component) -> exclusive seconds
    calls = collections.Counter()
    stack = []                                  # child time accumulated per open timer
    phase = "loop"


def wrap(fn, comp, phase=None):
    def w(*a, **k):
        prev = T.phase
        if phase is not None:
            T.phase = phase(a) if callable(phase) else phase
        cur = T.phase
        T.stack.append(0.0)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            el = time.perf_counter() - t0
            child = T.stack.pop()
            T.acc[(cur, comp)] += el - child
            if T.stack:
                T.stack[-1] += el
            T.phase = prev
            T.calls[comp] += 1
    w.__wrapped__ = fn
    return w


def instrument_fuzzer(f):
    """attach timers to one fuzzer instance (its model / executor / data generator / analyser)"""
    import rvzr.arch.x86.fuzzer as xf
    for m in (f.model, getattr(f, "arch_model", None)):
        if m is None:
            continue
        for name in ("load_test_case", "trace_test_case", "trace_test_case_with_taints"):
            setattr(m, name, wrap(getattr(m, name), "model"))
    for ex in (f.executor, getattr(f, "arch_executor", None)):
        if ex is None:
            continue
        for name in ("load_test_case", "trace_test_case"):
            setattr(ex, name, wrap(getattr(ex, name), "hardware"))
    f.data_gen.generate = wrap(f.data_gen.generate, "self", "inputs")
    f.data_gen.generate_boosted = wrap(f.data_gen.generate_boosted, "boost")
    for name in ("filter_violations", "htraces_are_equivalent"):
        setattr(f.analyser, name, wrap(getattr(f.analyser, name), "analysis"))
    f._filter = wrap(f._filter, "self", "filter")
    f._store_violation_artifact = wrap(f._store_violation_artifact, "self", "report")
    f.log.report_violations = wrap(f.log.report_violations, "self", "report")
    f.log.dbg_violation = wrap(f.log.dbg_violation, "self", "report")
    f.code_gen.create_test_case = wrap(f.code_gen.create_test_case, "self", "generate")
    if not getattr(xf._create_fenced_test_case, "__wrapped__", None):
        xf._create_fenced_test_case = wrap(xf._create_fenced_test_case, "assemble")


def instrument_classes():
    import importlib
    rf = importlib.import_module("rvzr.fuzzer")      # (the rvzr package's `fuzzer` attribute can
                                                     # be rebound to rvzr.arch.x86.fuzzer)
    if not getattr(rf._RoundManager.execute_stage, "__wrapped__", None):
        rf._RoundManager.execute_stage = wrap(
            rf._RoundManager.execute_stage, "self",
            lambda a: "fast" if a[1] == "fast" else f"confirm:{a[1]}")


def hook_overhead(n=200000):
    """seconds per timer call (wrapper cost), measured on a no-op"""
    def f():
        return None
    w = wrap(f, "probe")
    saved = (dict(T.acc), T.calls.copy())
    t0 = time.perf_counter()
    for _ in range(n):
        w()
    per = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    per -= (time.perf_counter() - t0) / n
    T.acc.clear()
    T.acc.update(saved[0])
    T.calls = saved[1]
    return per


def dump(out, extra):
    res = {"acc": {f"{p}|{c}": v for (p, c), v in T.acc.items()}, "calls": dict(T.calls), **extra}
    json.dump(res, open(os.path.join(out, "time_breakdown.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "acc"}, indent=1))


def run_revizor(out, n):
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    import rvzr.cli as cli
    from rvzr.stats import FuzzingStats
    instrument_classes()
    orig = cli.get_fuzzer
    box = {}

    def get_fuzzer(*a, **k):
        f = orig(*a, **k)
        instrument_fuzzer(f)
        start = f.start

        def timed_start(*sa, **sk):
            t0 = time.perf_counter()
            try:
                return start(*sa, **sk)
            finally:
                box["wall"] = time.perf_counter() - t0
        f.start = timed_start
        return f
    cli.get_fuzzer = get_fuzzer
    sys.argv = ["revizor.py", "fuzz", "-s", "base.json", "-c", CFG, "-I", os.path.dirname(CFG),
                "-n", str(n), "-i", "50", "-w", out, "--nonstop"]
    cli.main()
    st = FuzzingStats()
    stats = {k: getattr(st, k) for k in ("test_cases", "spec_filter", "observ_filter", "fast_path",
                                          "fp_nesting", "fp_taint_mistakes", "fp_priming",
                                          "fp_large_sample", "violations") if hasattr(st, k)}
    dump(out, {"tool": "revizor", "programs": stats.get("test_cases", n), "wall": box["wall"],
               "stats": stats, "hook_s_per_call": hook_overhead()})


class Snap:
    """per-program record, taken when the program's outcome goes back to the agent"""
    last_acc, last_t, i, f = {}, None, 0, None


def snap_after(fn):
    def w(*a, **k):
        r = fn(*a, **k)
        now = time.perf_counter()
        cur = dict(T.acc)
        delta = {}
        for key, v in cur.items():
            d = v - Snap.last_acc.get(key, 0.0)
            if d > 0:
                delta[key] = d
        Snap.last_acc = cur
        wall = now - Snap.last_t
        Snap.last_t = now
        g = group_of(delta, wall)
        Snap.f.write(json.dumps({"i": Snap.i, "stage": a[0], "wall": round(wall, 6),
                                 "g": {k: round(v, 6) for k, v in g.items() if v > 0}}) + "\n")
        Snap.i += 1
        return r
    return w


def run_rl(out, n, init, arch="gru"):
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, HERE)
    import tcenv
    instrument_classes()
    import rvzr.factory as fac
    orig_pg = fac.get_program_generator

    def get_program_generator(*a, **k):
        g = orig_pg(*a, **k)
        g.create_test_case = wrap(g.create_test_case, "self", "generate")
        return g
    fac.get_program_generator = get_program_generator
    orig_init = tcenv.TCEnv.__init__
    box = {}

    def init_(self, *a, **k):
        orig_init(self, *a, **k)
        instrument_fuzzer(self.fuzzer)
        if self.actor is not None:
            self.actor.before_generate = wrap(self.actor.before_generate, "self", "policy")
            self.actor.observe = snap_after(wrap(self.actor.observe, "self", "policy"))
            # get_nowait() calls get(False): only a blocking get (nothing ready) is a wait
            q_block = wrap(self.actor.prog_q.get, "wait")
            q_poll = wrap(self.actor.prog_q.get, "poll")

            def qget(*qa, **qk):
                block = qa[0] if qa else qk.get("block", True)
                return (q_block if block else q_poll)(*qa, **qk)
            self.actor.prog_q.get = qget
            Snap.f = open(os.path.join(out, "programs.jsonl"), "w", buffering=1 << 16)
        if getattr(self, "fx", None) is not None:
            fx = self.fx
            self.fx = wrap(fx, "self", "rl-harness")
            self.fx.names = fx.names                 # tcenv reads the feature names
        self._info = wrap(self._info, "self", "rl-harness")
        box["env"] = self
    tcenv.TCEnv.__init__ = init_
    orig_run = tcenv.TCEnv.run

    def run(self, *a, **k):
        t0 = time.perf_counter()
        Snap.last_t = t0
        try:
            return orig_run(self, *a, **k)
        finally:
            box["wall"] = time.perf_counter() - t0
    tcenv.TCEnv.run = run
    sys.argv = ["tcenv.py", "--mode", "bypass", "--minutes", "600", "--max-programs", str(n),
                "--out", out, "--action", json.dumps(ACTION), "--policy", "ctx",
                "--ctx-policy", "learn", "--ctx-mode", "rule2", "--ctx-arch", arch] + \
        (["--ctx-init", init] if init else [])
    tcenv.main()
    if Snap.f:
        Snap.f.close()
    s = json.load(open(os.path.join(out, "summary.json")))
    dump(out, {"tool": "rl", "arch": arch, "init": init, "programs": s["test_cases"], "wall": box["wall"],
               "stats": s["stages"], "hook_s_per_call": hook_overhead()})


# ----------------------------------------------------------------------------------- report
GROUPS = [   # (label, predicate on (phase, component))
    ("Program generation", lambda p, c: p == "generate"),
    ("Policy: waiting for agent", lambda p, c: p == "policy" and c == "wait"),
    ("Policy (RL)", lambda p, c: p == "policy"),
    ("Input generation", lambda p, c: p == "inputs"),
    ("Filters: hardware", lambda p, c: p == "filter" and c == "hardware"),
    ("Filters: other", lambda p, c: p == "filter" and c != "hardware"),
    ("Fast path: model", lambda p, c: p == "fast" and c in ("model", "boost")),
    ("Fast path: hardware", lambda p, c: p == "fast" and c == "hardware"),
    ("Fast path: other", lambda p, c: p == "fast" and c not in ("model", "boost", "hardware")),
    ("Confirmation", lambda p, c: p.startswith("confirm")),
    ("Reporting", lambda p, c: p == "report"),
    ("RL harness", lambda p, c: p == "rl-harness"),
    ("Other", lambda p, c: True),
]


def group_of(acc, wall=None):
    """{(phase, comp) or "phase|comp": seconds} -> {group: seconds}; wall: add the uncovered rest"""
    out = collections.OrderedDict((g, 0.0) for g, _ in GROUPS)
    for key, v in acc.items():
        p, c = key.split("|") if isinstance(key, str) else key
        for g, pred in GROUPS:
            if pred(p, c):
                out[g] += v
                break
    if wall is not None:
        out["Other"] += max(wall - sum(acc.values()), 0.0)
    return out


def groups(res):
    acc = {tuple(k.split("|")): v for k, v in res["acc"].items()}
    covered = sum(acc.values())
    acc[("loop", "self")] = acc.get(("loop", "self"), 0.0) + max(res["wall"] - covered, 0.0)
    out = collections.OrderedDict((g, 0.0) for g, _ in GROUPS)
    for (p, c), v in acc.items():
        for g, pred in GROUPS:
            if pred(p, c):
                out[g] += v
                break
    return out, acc


COL = {"Program generation": "#2a78d6", "Policy: waiting for agent": "#d95926", "Policy (RL)": "#eb6834",
       "Input generation": "#9085e9", "Filters: hardware": "#1baf7a", "Filters: other": "#8fd9be",
       "Fast path: model": "#6250d6", "Fast path: hardware": "#008300", "Fast path: other": "#b8b6ad",
       "Confirmation": "#eda100", "Reporting": "#e87ba4", "RL harness": "#d55181", "Other": "#d6d5d0"}
TEXT, TEXT2 = "#0b0b0b", "#52514e"


def _style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(TEXT2)
    ax.tick_params(colors=TEXT2)


def nviol(r):
    return r["stats"].get("violations", r["stats"].get("violation", 0))


def learner(d):
    """PPO updates from agent.jsonl and the parameter count from agent_header.json"""
    pa, ph = os.path.join(d, "agent.jsonl"), os.path.join(d, "agent_header.json")
    if not os.path.exists(pa):
        return None
    A = [json.loads(l) for l in open(pa) if l.strip()]
    u = [x["upd_s"] for x in A]
    h = json.load(open(ph)) if os.path.exists(ph) else {}
    return {"updates": len(u), "upd_mean": sum(u) / len(u) if u else 0, "upd_max": max(u) if u else 0,
            "upd_total": sum(u), "params": h.get("params"), "arch": h.get("arch"),
            "kl_last": A[-1]["kl"] if A else None}


def report(runs, fig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = [json.load(open(os.path.join(d, "time_breakdown.json"))) for _, d in runs]
    names = [n for n, _ in runs]
    G = [groups(r)[0] for r in R]
    lines = []
    for (name, d), r, g in zip(runs, R, G):
        n, v = r["programs"], nviol(r)
        conf = sum(val for (p, c), val in groups(r)[1].items() if p.startswith("confirm"))
        hook = r["hook_s_per_call"] * sum(r["calls"].values())
        lines.append(f"{name}: {n} programs, {v} violations ({1000 * v / n:.1f}/1000), wall "
                     f"{r['wall']:.1f} s ({r['wall'] / n * 1000:.1f} ms/program, {v / r['wall'] * 3600:.1f} V/h); "
                     f"confirmation {conf:.1f} s = {100 * conf / r['wall']:.1f}% = {conf / max(v, 1):.1f} s per "
                     f"violation; timer overhead {hook:.2f} s ({100 * hook / r['wall']:.2f}%)")
        if "wait" in r["calls"]:
            lines.append(f"    fuzzer blocked on the agent: {r['calls']['wait']} times, "
                         f"{groups(r)[1].get(('policy', 'wait'), 0):.2f} s")
        L = learner(d)
        if L:
            lines.append(f"    learner ({L['arch']}, {L['params']} params): {L['updates']} PPO updates, "
                         f"{L['upd_mean']:.2f} s each (max {L['upd_max']:.2f}), {L['upd_total']:.0f} s total "
                         f"= {100 * L['upd_total'] / r['wall']:.1f}% of wall (cores 4-7, off the critical path)")
        for k, val in g.items():
            if val > 0:
                lines.append(f"    {k:26s} {val / n * 1000:9.2f} ms/program  {100 * val / r['wall']:5.1f}%")
    txt = "\n".join(lines)
    print(txt)
    if not fig:
        return
    open(fig + ".txt", "w").write(txt + "\n")
    with open(fig + ".csv", "w") as f:
        f.write("component," + ",".join(f"{n} ms/program,{n} share" for n in names) + "\n")
        for k in G[0]:
            f.write(k + "," + ",".join(f"{g[k] / r['programs'] * 1000:.3f},{g[k] / r['wall']:.4f}"
                                       for g, r in zip(G, R)) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig_, ax = plt.subplots(figsize=(6.4, 1.6 + 0.9 * len(R)), dpi=200)
    fig_.patch.set_facecolor("white")
    left = [0.0] * len(R)
    for k in G[0]:
        vals = [g[k] / r["programs"] * 1000 for g, r in zip(G, R)]
        if max(vals) <= 0:
            continue
        ax.barh(names, vals, left=left, color=COL[k], edgecolor="white", linewidth=1.5, height=0.5,
                label=k)
        left = [a + b for a, b in zip(left, vals)]
    for i, l in enumerate(left):
        ax.text(l, i, f"  {l:.0f} ms", va="center", color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock time per program (ms)", color=TEXT)
    _style(ax)
    ax.set_xlim(0, max(left) * 1.15)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig_.suptitle("Spectre v4: time breakdown per program", x=0.02, ha="left", fontsize=11,
                  fontweight="bold", color=TEXT)
    fig_.tight_layout()
    fig_.savefig(fig + ".png", bbox_inches="tight")
    fig_.savefig(fig + ".pdf", bbox_inches="tight")
    print(fig + ".png")


MERGE = [("Generation + policy + inputs", ("Program generation", "Policy: waiting for agent", "Policy (RL)",
                                          "Input generation"), "#2a78d6"),
         ("Filters", ("Filters: hardware", "Filters: other"), "#1baf7a"),
         ("Fast path", ("Fast path: model", "Fast path: hardware", "Fast path: other"), "#6250d6"),
         ("Confirmation", ("Confirmation",), "#eda100"),
         ("Other (RL harness, logging)", ("Reporting", "RL harness", "Other"), "#b8b6ad")]


def blocks(runs, fig, blk=1000):
    """per block of `blk` programs: ms/program per (merged) group, confirmation share, violation rate"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out, lines = [], []
    for name, d in runs:
        P = [json.loads(l) for l in open(os.path.join(d, "programs.jsonl")) if l.strip()]
        rows = []
        for b in range(0, len(P), blk):
            X = P[b:b + blk]
            wall = sum(x["wall"] for x in X)
            m = {lab: sum(x["g"].get(k, 0) for x in X for k in ks) / len(X) * 1000 for lab, ks, _ in MERGE}
            conf = m["Confirmation"] * len(X) / 1000
            v = sum(x["stage"] == "violation" for x in X)
            rows.append({"block": f"{b + 1}-{b + len(X)}", "n": len(X), "ms": m, "ms_total": wall / len(X) * 1000,
                         "conf_share": conf / wall if wall else 0, "v_per_1000": 1000 * v / len(X)})
        out.append((name, rows))
        lines.append(f"{name} (per {blk} programs): ms/program [" + ", ".join(lab for lab, _, _ in MERGE) +
                     "] | total ms | confirmation share | violations per 1000")
        for r in rows:
            lines.append(f"  {r['block']:>11s}  " + " ".join(f"{r['ms'][lab]:7.1f}" for lab, _, _ in MERGE) +
                         f"  | {r['ms_total']:7.1f} | {100 * r['conf_share']:5.1f}% | {r['v_per_1000']:5.1f}")
    txt = "\n".join(lines)
    print(txt)
    if not fig:
        return
    open(fig + ".txt", "w").write(txt + "\n")
    with open(fig + ".csv", "w") as f:
        f.write("run,block,n," + ",".join(f"{lab} ms/program" for lab, _, _ in MERGE) +
                ",total ms/program,confirmation share,violations per 1000\n")
        for name, rows in out:
            for r in rows:
                f.write(f"{name},{r['block']},{r['n']}," + ",".join(f"{r['ms'][lab]:.3f}" for lab, _, _ in MERGE) +
                        f",{r['ms_total']:.3f},{r['conf_share']:.4f},{r['v_per_1000']:.2f}\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig_, axes = plt.subplots(2, len(out), figsize=(3.4 * len(out), 4.6), dpi=200, sharex="col",
                              gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12}, squeeze=False)
    fig_.patch.set_facecolor("white")
    ymax = max(r["ms_total"] for _, rows in out for r in rows) * 1.18
    vmax = max(r["v_per_1000"] for _, rows in out for r in rows) * 1.25
    for j, (name, rows) in enumerate(out):
        ax, axv = axes[0][j], axes[1][j]
        x = list(range(len(rows)))
        bottom = [0.0] * len(rows)
        for lab, _, col in MERGE:
            vals = [r["ms"][lab] for r in rows]
            ax.bar(x, vals, bottom=bottom, color=col, edgecolor="white", linewidth=1.0, width=0.75,
                   label=lab if j == 0 else None)
            bottom = [a + b for a, b in zip(bottom, vals)]
        for xi, r in zip(x, rows):
            ax.text(xi, bottom[xi], f"{100 * r['conf_share']:.0f}%", ha="center", va="bottom",
                    fontsize=7, color=TEXT2)
        ax.set_title(name, loc="left", fontsize=10, color=TEXT, fontweight="bold")
        ax.set_ylim(0, ymax)
        axv.bar(x, [r["v_per_1000"] for r in rows], color="#eb6834", width=0.75)
        axv.set_ylim(0, vmax)
        axv.set_xticks(x)
        axv.set_xticklabels([f"{(i + 1) * blk // 1000}k" for i in x], fontsize=7)
        axv.set_xlabel("Programs generated", color=TEXT)
        if j == 0:
            ax.set_ylabel("ms per program", color=TEXT)
            axv.set_ylabel("Violations\nper 1000", color=TEXT)
        for a_ in (ax, axv):
            _style(a_)
    fig_.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig_.suptitle("Spectre v4: time per program over training (label: confirmation share)",
                  x=0.02, ha="left", fontsize=10.5, fontweight="bold", color=TEXT)
    fig_.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.2, wspace=0.18)
    fig_.savefig(fig + ".png", bbox_inches="tight")
    fig_.savefig(fig + ".pdf", bbox_inches="tight")
    print(fig + ".png")


def latency(runs, reps=200):
    """agent-side sampling latency of one FRESH program (batch 1), final policy of each run"""
    import numpy as np
    sys.path.insert(0, HERE)
    import torch
    torch.set_num_threads(4)
    import ctxrl
    import progrl
    _, table, _, _ = progrl._offline_generator(ACTION)
    for name, d in runs:
        h = json.load(open(os.path.join(d, "agent_header.json")))
        pol = ctxrl.make_policy(table, arch=h["arch"], **(h.get("tf") or {}))
        pol.load_state_dict(torch.load(os.path.join(d, "policy.pt")))
        pol.eval()
        sm = ctxrl.CtxSampler(table, pol, np.random.default_rng(0))
        c0 = ctxrl.ctx_vector(0, 8, None)
        for _ in range(5):
            sm.sample(c0[None], [None])
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _, prog, toks, _ = sm.sample(c0[None], [None])[0]
            t.append(time.perf_counter() - t0)
        t.sort()
        print(f"{name} ({h['arch']}, {h.get('params')} params): sampling one program ({len(toks)} decisions) "
              f"median {1000 * t[len(t) // 2]:.1f} ms, p90 {1000 * t[int(0.9 * len(t))]:.1f} ms")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("report", "blocks", "latency"):
        a = sys.argv[2:]
        fig = a[a.index("--fig") + 1] if "--fig" in a else None
        blk = int(a[a.index("--block") + 1]) if "--block" in a else 1000
        runs = [x for x in a if "=" in x]
        if not runs:                                       # old form: <revizor dir> <rl dir>
            runs = [f"Revizor={a[0]}", f"RL (GRU)={a[1]}"]
        runs = [tuple(x.split("=", 1)) for x in runs]
        if cmd == "report":
            report(runs, fig)
        elif cmd == "blocks":
            blocks(runs, fig, blk)
        else:
            latency(runs)
    else:
        out = os.path.abspath(sys.argv[2])
        os.makedirs(out, exist_ok=True)
        n = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 3000
        if cmd == "revizor":
            run_revizor(out, n)
        else:
            init = sys.argv[sys.argv.index("--init") + 1] if "--init" in sys.argv else WARM
            arch = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else "gru"
            run_rl(out, n, None if init == "none" else init, arch)
