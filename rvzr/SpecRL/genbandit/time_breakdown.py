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
  taskset -c 1-3 python rvzr/SpecRL/genbandit/time_breakdown.py rl <out> [N] [--init policy.pt]
  python rvzr/SpecRL/genbandit/time_breakdown.py report <revizor out> <rl out> --fig <path prefix>
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


def run_rl(out, n, init):
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
            self.actor.observe = wrap(self.actor.observe, "self", "policy")
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
        try:
            return orig_run(self, *a, **k)
        finally:
            box["wall"] = time.perf_counter() - t0
    tcenv.TCEnv.run = run
    sys.argv = ["tcenv.py", "--mode", "bypass", "--minutes", "600", "--max-programs", str(n),
                "--out", out, "--action", json.dumps(ACTION), "--policy", "ctx",
                "--ctx-policy", "learn", "--ctx-mode", "rule2"] + (["--ctx-init", init] if init else [])
    tcenv.main()
    s = json.load(open(os.path.join(out, "summary.json")))
    dump(out, {"tool": "rl", "init": init, "programs": s["test_cases"], "wall": box["wall"],
               "stats": s["stages"], "hook_s_per_call": hook_overhead()})


# ----------------------------------------------------------------------------------- report
GROUPS = [   # (label, predicate on (phase, component))
    ("Program generation", lambda p, c: p == "generate"),
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


def report(rv_dir, rl_dir, fig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = [json.load(open(os.path.join(d, "time_breakdown.json"))) for d in (rv_dir, rl_dir)]
    names = ["Revizor", "RL (GRU)"]
    G = [groups(r)[0] for r in R]
    lines = []
    for name, r, g in zip(names, R, G):
        n, v = r["programs"], r["stats"].get("violations", r["stats"].get("violation", 0))
        conf = sum(val for (p, c), val in groups(r)[1].items() if p.startswith("confirm"))
        hook = r["hook_s_per_call"] * sum(r["calls"].values())
        lines.append(f"{name}: {n} programs, {v} violations, wall {r['wall']:.1f} s "
                     f"({r['wall'] / n * 1000:.1f} ms/program, {v / r['wall'] * 3600:.1f} V/h); "
                     f"confirmation {conf:.1f} s = {conf / max(v, 1):.1f} s per violation; "
                     f"timer overhead {hook:.2f} s ({100 * hook / r['wall']:.2f}%)")
        for k, val in g.items():
            if val > 0:
                lines.append(f"    {k:22s} {val / n * 1000:9.2f} ms/program  {100 * val / r['wall']:5.1f}%")
    txt = "\n".join(lines)
    print(txt)
    open(fig + ".txt", "w").write(txt + "\n")
    with open(fig + ".csv", "w") as f:
        f.write("component," + ",".join(f"{n} ms/program,{n} share" for n in names) + "\n")
        for k in G[0]:
            f.write(k + "," + ",".join(f"{g[k] / r['programs'] * 1000:.3f},{g[k] / r['wall']:.4f}"
                                       for g, r in zip(G, R)) + "\n")
    COL = {"Program generation": "#2a78d6", "Policy (RL)": "#eb6834", "Input generation": "#9085e9",
           "Filters: hardware": "#1baf7a", "Filters: other": "#8fd9be",
           "Fast path: model": "#6250d6", "Fast path: hardware": "#008300", "Fast path: other": "#b8b6ad",
           "Confirmation": "#eda100", "Reporting": "#e87ba4", "RL harness": "#d55181", "Other": "#d6d5d0"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig_, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    fig_.patch.set_facecolor("white")
    ax.set_facecolor("white")
    left = [0.0, 0.0]
    for k in G[0]:
        vals = [g[k] / r["programs"] * 1000 for g, r in zip(G, R)]
        if max(vals) <= 0:
            continue
        ax.barh(names, vals, left=left, color=COL[k], edgecolor="white", linewidth=1.5, height=0.5,
                label=k)
        left = [a + b for a, b in zip(left, vals)]
    for i, (r, l) in enumerate(zip(R, left)):
        ax.text(l, i, f"  {l:.0f} ms", va="center", color="#0b0b0b", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock time per program (ms)", color="#0b0b0b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#52514e")
    ax.tick_params(colors="#52514e")
    ax.set_xlim(0, max(left) * 1.15)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig_.suptitle("Spectre v4: time breakdown per program", x=0.02, ha="left", fontsize=11,
                  fontweight="bold", color="#0b0b0b")
    fig_.tight_layout()
    fig_.savefig(fig + ".png", bbox_inches="tight")
    fig_.savefig(fig + ".pdf", bbox_inches="tight")
    print(fig + ".png")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "report":
        a = sys.argv[2:]
        report(a[0], a[1], a[a.index("--fig") + 1])
    else:
        out = os.path.abspath(sys.argv[2])
        os.makedirs(out, exist_ok=True)
        n = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 3000
        if cmd == "revizor":
            run_revizor(out, n)
        else:
            init = sys.argv[sys.argv.index("--init") + 1] if "--init" in sys.argv else WARM
            run_rl(out, n, init)
