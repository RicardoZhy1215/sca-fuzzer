"""
Coupled knob policy for the v4 space: the knobs are chosen JOINTLY, per test case, from a model
that has pairwise knob interactions -- instead of one independent softmax per knob (rl.py).

Model (one row per distinct configuration c, fitted on aggregated per-test-case outcomes):
  logit P(violation | c) = b + sum_k w[c_k] + sum_{j<k} <u[c_j], u[c_k]>
  log E[seconds | c, no violation] = the same form, separate parameters
  V/h(c) = 3600 * p(c) / (t(c) + p(c) * T_v)          (T_v = seconds a violation costs)
Every knob value has a weight w and an embedding u (rank R); two knobs interact through the dot
product of their embeddings (a factorization machine; same building block as SpecRL's embedding
heads, but small enough to learn from a few hundred violations). value_mix=full64 makes
entropy_bits irrelevant, so its entropy one-hot is zeroed (a structural mask) and full64
configurations are enumerated once.

Policy (tcenv.py --policy coupled; this file's `learn` loop runs as a separate process on
cores 4-7 and talks to tcenv through files):
  - with prob 1 - alpha: the incumbent configuration (starts at the frozen best of the v4 RL run)
  - with prob alpha: half the time the posterior-mean best configuration (the promotion
    candidate), half the time the best configuration of one random bootstrap member
    (Thompson sampling over K members fitted to Poisson-resampled counts)
  - the incumbent changes only on LIVE interleaved evidence: a configuration with >= MIN_TC
    live test cases whose 95% lower bound on V/h beats the incumbent's live mean takes over
    (the same rule rolls back a lucky promotion)
Warm start: per-configuration counts from the v4 RL slices and the step-2 collection.

Every round the learner writes policy.json (read by tcenv) and insight.txt / insight.json:
which knob pairs interact, how strongly (bootstrap range), the best configuration with and
without interactions.

  python rvzr/SpecRL/genbandit/coupled.py selftest          # synthetic world, planted interaction
  python rvzr/SpecRL/genbandit/coupled.py insight --data <run dirs or slices.jsonl ...>
  (learn is started by tcenv.py)
"""
import argparse
import itertools
import json
import math
import os
import random
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import space_v4 as space  # noqa: E402

KN = list(space.KNOBS)
SIZES = [len(space.KNOBS[k]) for k in KN]
OFF = np.cumsum([0] + SIZES[:-1])
D = int(sum(SIZES))
I_MIX, I_ENT = KN.index("value_mix"), KN.index("entropy_bits")
FULL64 = space.KNOBS["value_mix"].index("full64")
ENT_CANON = space.KNOBS["entropy_bits"].index(10)
T_V_PRIOR = 10.7          # seconds per violation (step-2 collection)
MIN_TC = 1500             # live test cases before a configuration can become the incumbent
RANK = 4
LAM_W, LAM_U = 0.125, 2.0  # Gaussian priors: sd 2 on main effects, sd 0.5 on embeddings
WARM = [os.path.join(HERE, "runs", "v4-rl-261103-0208", "slices.jsonl"),
        os.path.join(HERE, "runs", "tc-collect-261103-2246")]


# ----------------------------------------------------------------------------- configurations
def canon(idx):
    idx = list(idx)
    if idx[I_MIX] == FULL64:
        idx[I_ENT] = ENT_CANON
    return tuple(int(i) for i in idx)


def to_idx(a: dict):
    return canon(space.KNOBS[k].index(a[k]) for k in KN)


def to_action(idx) -> dict:
    return {k: space.KNOBS[k][i] for k, i in zip(KN, idx)}


def encode(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, np.int64).reshape(-1, len(KN))
    X = np.zeros((len(rows), D), np.float32)
    for j in range(len(KN)):
        X[np.arange(len(rows)), OFF[j] + rows[:, j]] = 1
    X[rows[:, I_MIX] == FULL64, OFF[I_ENT]:OFF[I_ENT] + SIZES[I_ENT]] = 0
    return X


def all_configs() -> np.ndarray:
    return np.array(sorted({canon(t) for t in itertools.product(*[range(s) for s in SIZES])}),
                    np.int64)


# ----------------------------------------------------------------------------- model
def fm_predict(p, X):
    b, w, U = p
    XU = X @ U
    return b + X @ w + 0.5 * ((XU ** 2) - X @ (U ** 2)).sum(1)


def fm_fit(X, kind, y1, y2, rank=RANK, init=None, seed=0, iters=150, anchor=None):
    """kind 'binom': y1 = positives, y2 = negatives per row. kind 'gauss': y1 = target,
    y2 = weight. Returns (b, w, U) as numpy. Bias unpenalized; Gaussian priors centered on
    `anchor` = (w0, U0) (zero if None). A random anchor drawn from the prior turns the MAP fit
    into one posterior sample (randomized prior / anchored ensemble), which is what makes the
    ensemble disagree -- and therefore explore -- where there is no data."""
    import torch
    torch.manual_seed(seed)
    Xt = torch.tensor(X)
    a1, a2 = torch.tensor(y1, dtype=torch.float32), torch.tensor(y2, dtype=torch.float32)
    if init is None:
        b0 = 0.0
        if kind == "binom":
            pr = (a1.sum() + 0.5) / (a1.sum() + a2.sum() + 1)
            b0 = float(torch.log(pr / (1 - pr)))
        else:
            b0 = float((a1 * a2).sum() / a2.sum())
        init = (b0, np.zeros(D, np.float32), 0.01 * np.random.default_rng(seed).standard_normal(
            (D, rank)).astype(np.float32))
    b = torch.tensor([float(init[0])], requires_grad=True)
    w = torch.tensor(init[1], requires_grad=True)
    U = torch.tensor(init[2], requires_grad=True)
    w0 = torch.zeros(D) if anchor is None else torch.tensor(anchor[0])
    U0 = torch.zeros(D, rank) if anchor is None else torch.tensor(anchor[1][:, :rank])
    params = [b, w] + ([U] if rank else [])
    opt = torch.optim.LBFGS(params, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        XU = Xt @ U
        z = b + Xt @ w + (0.5 * ((XU ** 2) - Xt @ (U ** 2)).sum(1) if rank else 0)
        if kind == "binom":
            nll = (a1 * torch.nn.functional.softplus(-z) + a2 * torch.nn.functional.softplus(z)).sum()
        else:
            nll = 0.5 * (a2 * (z - a1) ** 2).sum()
        loss = nll + LAM_W * ((w - w0) ** 2).sum() + LAM_U * ((U - U0) ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return (float(b.detach()), w.detach().numpy().copy(), U.detach().numpy().copy())


# ----------------------------------------------------------------------------- data
class Data:
    """Per-configuration counts: n test cases, V violations, seconds (all), seconds of
    non-violating test cases; violation seconds per n_inputs index (tcenv data only)."""

    def __init__(self):
        self.c = {}
        self.tv = {}

    def add(self, idx, n, v, sec, sec_nv, n_nv):
        r = self.c.setdefault(idx, [0, 0, 0.0, 0.0, 0])
        r[0] += n
        r[1] += v
        r[2] += sec
        r[3] += sec_nv
        r[4] += n_nv

    def add_tc(self, rec):
        idx = canon(rec["a"])
        v = rec["stage"] == "violation"
        self.add(idx, 1, int(v), rec["s"], 0.0 if v else rec["s"], 0 if v else 1)
        if v:
            self.tv.setdefault(idx[KN.index("n_inputs")], []).append(rec["s"])

    def merged(self, other):
        m = Data()
        for d in (self, other):
            for k, r in d.c.items():
                m.add(k, *r)
            for k, l in d.tv.items():
                m.tv.setdefault(k, []).extend(l)
        return m

    def arrays(self):
        keys = list(self.c)
        R = np.array([self.c[k] for k in keys], np.float64)
        return np.array(keys, np.int64), R

    def t_v(self):
        allv = [s for l in self.tv.values() for s in l]
        pooled = float(np.mean(allv)) if len(allv) >= 5 else T_V_PRIOR
        return np.array([float(np.mean(self.tv[i])) if len(self.tv.get(i, [])) >= 5 else pooled
                         for i in range(SIZES[KN.index("n_inputs")])])


class TcTail:
    """Reads complete lines appended to a tc.jsonl since the last call."""

    def __init__(self, path):
        self.path, self.pos, self.buf = path, 0, ""

    def read(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            f.seek(self.pos)
            chunk = f.read()
            self.pos = f.tell()
        self.buf += chunk
        lines = self.buf.split("\n")
        self.buf = lines.pop()
        out = []
        for l in lines:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
        return out


def load_warm(paths) -> Data:
    d = Data()
    for p in paths:
        if p.endswith(".jsonl") and "slices" in os.path.basename(p):
            for l in open(p):
                s = json.loads(l)
                if s.get("test_cases", 0) <= 0:
                    continue
                sec = s["hours"] * 3600
                nv = s["test_cases"] - s["violations"]
                d.add(to_idx(s["action"]), s["test_cases"], s["violations"], sec,
                      max(sec - s["violations"] * T_V_PRIOR, 0.05 * nv), nv)
        else:
            for rec in TcTail(os.path.join(p, "tc.jsonl")).read():
                d.add_tc(rec)
    return d


# ----------------------------------------------------------------------------- learner
def poisson_lcb(v, conf=0.95):
    from scipy.stats import chi2
    return 0.0 if v <= 0 else float(chi2.ppf(1 - conf, 2 * v) / 2)


class Learner:
    def __init__(self, k_members=16, rank=RANK, seed=0):
        self.K, self.rank = k_members, rank
        self.rng = np.random.default_rng(seed)
        self.C = all_configs()
        self.XC = encode(self.C)
        self.members, self.tmodel, self.main_only = [None] * k_members, None, None
        sd_w, sd_u = 1 / math.sqrt(2 * LAM_W), 1 / math.sqrt(2 * LAM_U)
        self.anchors = [None] + [(self.rng.normal(0, sd_w, D).astype(np.float32),
                                  self.rng.normal(0, sd_u, (D, rank)).astype(np.float32))
                                 for _ in range(k_members - 1)]

    def fit(self, data: Data):
        keys, R = data.arrays()
        X = encode(keys)
        n, v = R[:, 0], R[:, 1]
        mean_member = None
        for k in range(self.K):
            if k == 0:
                pos, neg = v, n - v
            else:
                pos, neg = self.rng.poisson(v), self.rng.poisson(n - v)
            init = self.members[k]
            if init is None and self.anchors[k] is not None:
                init = (None, self.anchors[k][0], self.anchors[k][1])
            if init is not None and init[0] is None:   # bias from the data
                pr = (pos.sum() + 0.5) / (pos.sum() + neg.sum() + 1)
                init = (math.log(pr / (1 - pr)), init[1], init[2])
            self.members[k] = fm_fit(X, "binom", pos, neg, self.rank, init, seed=k,
                                     anchor=self.anchors[k])
            if k == 0:
                mean_member = self.members[0]
        self.main_only = fm_fit(X, "binom", v, n - v, 0, None)
        ok = R[:, 4] > 0
        y = np.log(R[ok, 3] / R[ok, 4])
        self.tmodel = fm_fit(X[ok], "gauss", y, np.minimum(R[ok, 4], 5000), self.rank,
                             self.tmodel, seed=99)
        self.Tv = data.t_v()
        return mean_member

    def vph(self, p, X=None, rows=None):
        X = self.XC if X is None else X
        rows = self.C if rows is None else rows
        pv = 1 / (1 + np.exp(-fm_predict(p, X)))
        t = np.exp(fm_predict(self.tmodel, X))
        tv = self.Tv[rows[:, KN.index("n_inputs")]]
        return 3600 * pv / (t + pv * tv)

    def scores(self):
        S = np.stack([self.vph(m) for m in self.members])     # K x configs
        return S

    def prior_floor(self):
        vals = []
        for j, k in itertools.combinations(range(len(KN)), 2):
            Ms = [a[1][OFF[j]:OFF[j] + SIZES[j]] @ a[1][OFF[k]:OFF[k] + SIZES[k]].T
                  for a in self.anchors[1:]]
            vals.append(float(np.mean(Ms, 0).std()))
        return float(np.median(vals)), float(np.percentile(vals, 90))

    def interactions(self):
        """Per knob pair: sd over the pair's value grid of the interaction log-odds, for the
        ensemble-mean interaction matrix (independent prior noise averages out), and the
        10-90% range of the same quantity over members."""
        out = {}
        for j, k in itertools.combinations(range(len(KN)), 2):
            Ms = [m[2][OFF[j]:OFF[j] + SIZES[j]] @ m[2][OFF[k]:OFF[k] + SIZES[k]].T
                  for m in self.members[1:]]
            vals = [float(M.std()) for M in Ms]
            out[(KN[j], KN[k])] = (float(np.mean(Ms, 0).std()), float(np.percentile(vals, 10)),
                                   float(np.percentile(vals, 90)))
        return out


def empirical(live: Data, idx):
    r = live.c.get(idx)
    if not r or r[2] <= 0:
        return None
    return {"n": r[0], "v": r[1], "vph": 3600 * r[1] / r[2], "lcb": 3600 * poisson_lcb(r[1]) / r[2]}


def choose_incumbent(live: Data, inc):
    """Switch to the configuration whose live 95% lower bound beats the incumbent's live mean."""
    cur = empirical(live, inc)
    if cur is None or cur["n"] < MIN_TC:
        return inc, None
    best, best_lcb = inc, cur["vph"]
    for idx, r in live.c.items():
        if idx == inc or r[0] < MIN_TC:
            continue
        e = empirical(live, idx)
        if e["lcb"] > best_lcb:
            best, best_lcb = idx, e["lcb"]
    return best, (None if best == inc else f"lcb {best_lcb:.1f} > incumbent mean {cur['vph']:.1f}")


def pick_candidate(L, S, cur):
    """Posterior-mean best, but sticky: the current candidate is replaced only if the new one
    is predicted >= 5% better on average AND by >= 80% of the members, so that the candidate
    keeps collecting the live test cases it needs to be promoted."""
    mean = S.mean(0)
    new = int(np.argmax(mean))
    if cur is None:
        return L.C[new]
    rows = {tuple(int(x) for x in r): i for i, r in enumerate(L.C)}
    ci = rows.get(tuple(cur))
    if ci is None or ci == new:
        return L.C[new]
    if mean[new] >= 1.05 * mean[ci] and (S[1:, new] > S[1:, ci]).mean() >= 0.8:
        return L.C[new]
    return L.C[ci]


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def insight_text(L: Learner, data: Data, live: Data, inc, S, extra=""):
    mean = S.mean(0)
    best = int(np.argmax(mean))
    fact_best = int(np.argmax(L.vph(L.main_only)))
    lo, hi = np.percentile(S[1:, best], [10, 90])
    keys, R = data.arrays()
    lines = [f"[coupled model] {time.strftime('%F %T')}  data: {len(keys)} configs, "
             f"{int(R[:, 0].sum())} tc, {int(R[:, 1].sum())} violations; live: "
             f"{sum(r[0] for r in live.c.values())} tc, {sum(r[1] for r in live.c.values())} V",
             f"T_v by n_inputs: {np.round(L.Tv, 1).tolist()}", extra]
    ii = list(map(tuple, L.C)).index(inc) if inc in set(map(tuple, L.C)) else None
    e = empirical(live, inc)
    lines.append(f"incumbent: {to_action(inc)}")
    if ii is not None:
        lines.append(f"  predicted {mean[ii]:.0f} V/h" + (f"; live {e['vph']:.0f} V/h "
                     f"(n={e['n']}, V={e['v']})" if e else ""))
    lines.append(f"best with interactions: {to_action(L.C[best])}")
    lines.append(f"  predicted {mean[best]:.0f} V/h (bootstrap 10-90%: {lo:.0f}-{hi:.0f})")
    lines.append(f"best without interactions (main effects only): {to_action(L.C[fact_best])}")
    lines.append(f"  predicted by the full model: {mean[fact_best]:.0f} V/h")
    inter = sorted(L.interactions().items(), key=lambda kv: -kv[1][0])
    floor = L.prior_floor()
    lines.append("knob interactions (sd of the ensemble-mean interaction log-odds; members 10-90%):")
    lines.append(f"  noise floor = the same statistic for the prior anchors alone (no data): median "
                 f"{floor[0]:.2f}, 90th percentile {floor[1]:.2f} over pairs")
    for (a, b), (m, l, h) in inter[:10]:
        lines.append(f"  {a:13s} x {b:13s} {m:5.2f}  [{l:4.2f}, {h:4.2f}]")
    # conditional table of the strongest pair, other knobs at the incumbent
    (a, b), _ = inter[0]
    ja, jb = KN.index(a), KN.index(b)
    lines.append(f"P(violation) per 1000 tc for {a} x {b}, other knobs at the incumbent "
                 f"(model, ensemble mean):")
    for va in range(SIZES[ja]):
        row = []
        for vb in range(SIZES[jb]):
            idx = list(inc)
            idx[ja], idx[jb] = va, vb
            p = np.mean([1 / (1 + np.exp(-fm_predict(m, encode(np.array([canon(idx)])))[0]))
                         for m in L.members[1:]])
            row.append(f"{space.KNOBS[b][vb]}:{1000 * p:6.2f}")
        lines.append(f"  {str(space.KNOBS[a][va]):8s} " + "  ".join(row))
    return "\n".join(lines), {"incumbent": to_action(inc), "best": to_action(L.C[best]),
                              "best_vph": float(mean[best]),
                              "factored_best": to_action(L.C[fact_best]),
                              "factored_best_vph": float(mean[fact_best]),
                              "interactions": {f"{a}|{b}": v for (a, b), v in inter}}


def learn(args):
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import torch
    torch.set_num_threads(4)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    warm = load_warm(args.warm)
    live = Data()
    tail = TcTail(os.path.join(args.run, "tc.jsonl"))
    L = Learner(args.members, seed=args.seed)
    inc = to_idx(json.loads(args.incumbent))
    hist = open(os.path.join(args.run, "learner.jsonl"), "a")
    rnd, cand = 0, None
    while True:
        last = os.path.exists(os.path.join(args.run, "summary.json")) or stop["flag"]
        for rec in tail.read():
            live.add_tc(rec)
        t0 = time.time()
        data = warm.merged(live)
        L.fit(data)
        S = L.scores()
        cand = tuple(int(x) for x in pick_candidate(L, S, cand))
        explore = [to_action(L.C[int(np.argmax(S[k]))]) for k in range(1, L.K)]
        new_inc, why = choose_incumbent(live, inc)
        if why:
            print(f"[coupled] incumbent -> {to_action(new_inc)} ({why})", flush=True)
            inc = new_inc
        write_json(os.path.join(args.run, "policy.json"),
                   {"incumbent": to_action(inc), "candidate": to_action(cand),
                    "explore": explore, "alpha": args.alpha, "cand_frac": args.cand_frac,
                    "round": rnd})
        txt, js = insight_text(L, data, live, inc, S, f"fit {time.time() - t0:.1f} s, round {rnd}")
        open(os.path.join(args.run, "insight.txt"), "w").write(txt + "\n")
        write_json(os.path.join(args.run, "insight.json"), js)
        hist.write(json.dumps({"round": rnd, "time": time.strftime("%F %T"), "fit_s": round(time.time() - t0, 2),
                               "incumbent": to_action(inc), "candidate": to_action(cand),
                               "live_tc": sum(r[0] for r in live.c.values()),
                               "live_v": sum(r[1] for r in live.c.values()),
                               "best_vph": js["best_vph"]}) + "\n")
        hist.flush()
        rnd += 1
        if last:
            break
        deadline = time.time() + args.interval
        while time.time() < deadline and not stop["flag"] and \
                not os.path.exists(os.path.join(args.run, "summary.json")):
            time.sleep(2)


# ----------------------------------------------------------------------------- self-test
def selftest(args):
    """Synthetic world with a planted interaction: binary=off helps only at entropy 4, and
    flags=on helps only at program_size 64. Checks that the learner ranks the planted pairs
    first and that the policy moves to the joint optimum the independent model misses."""
    import torch
    torch.set_num_threads(4)
    rng = np.random.default_rng(1)
    C = all_configs()
    XC = encode(C)
    ji = {k: KN.index(k) for k in KN}

    def true_logit(rows):
        rows = np.atleast_2d(rows)
        z = -6.0 + 0.6 * rows[:, ji["program_size"]] + 0.4 * rows[:, ji["mem_ratio"]]
        z += -0.3 * rows[:, ji["entropy_bits"]] * (rows[:, ji["value_mix"]] == 0)
        z += -0.8 * (rows[:, ji["value_mix"]] == FULL64)
        b_off = rows[:, ji["binary"]] == 1
        e4 = (rows[:, ji["entropy_bits"]] == 0) & (rows[:, ji["value_mix"]] == 0)
        z += np.where(b_off & e4, 1.2, np.where(b_off, -0.6, 0.0))
        f_on, p64 = rows[:, ji["flags"]] == 0, rows[:, ji["program_size"]] == 3
        z += np.where(f_on & p64, 0.8, np.where(f_on, -0.5, 0.0))
        return z

    def true_vph(rows):
        p = 1 / (1 + np.exp(-true_logit(rows)))
        t = 0.03 + 0.0006 * np.array(space.KNOBS["program_size"])[np.atleast_2d(rows)[:, 0]]
        return 3600 * p / (t + p * T_V_PRIOR)

    tv = true_vph(C)
    opt = int(np.argmax(tv))
    print(f"true optimum {to_action(C[opt])}: {tv[opt]:.0f} V/h")
    inc = to_idx({**space.CONTROL})
    live, L = Data(), Learner(8, seed=0)
    policy, cand = None, None
    for rnd in range(args.rounds):
        # one round = ~2 min of test cases (~800)
        for _ in range(800):
            if policy and rng.random() < 0.3:
                idx = to_idx(policy["candidate"] if rng.random() < 0.5 else
                             policy["explore"][rng.integers(len(policy["explore"]))])
            else:
                idx = inc
            p = 1 / (1 + math.exp(-true_logit(np.array(idx))[0]))
            v = rng.random() < p
            t = 0.03 + 0.0006 * space.KNOBS["program_size"][idx[0]]
            live.add_tc({"a": list(idx), "stage": "violation" if v else "observ_filter",
                         "s": T_V_PRIOR if v else t})
        L.fit(live)
        S = L.scores()
        cand = tuple(int(x) for x in pick_candidate(L, S, cand))
        policy = {"candidate": to_action(cand),
                  "explore": [to_action(L.C[int(np.argmax(S[k]))]) for k in range(1, L.K)]}
        inc, why = choose_incumbent(live, inc)
        if rnd % 5 == 4 or rnd == args.rounds - 1 or why:
            ii = list(map(tuple, C)).index(inc)
            print(f"round {rnd + 1}: incumbent true {tv[ii]:.0f} V/h ({100 * tv[ii] / tv[opt]:.0f}% "
                  f"of optimum); candidate true {true_vph(np.array(cand))[0]:.0f}, live n "
                  f"{live.c.get(cand, [0])[0]}" + (f"  PROMOTED ({why})" if why else ""))
    txt, _ = insight_text(L, live, live, inc, L.scores())
    print(txt)
    fb = L.C[int(np.argmax(L.vph(L.main_only)))]
    print(f"main-effects-only argmax, true V/h: {true_vph(fb)[0]:.0f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("learn")
    a.add_argument("--run", required=True)
    a.add_argument("--incumbent", required=True)
    a.add_argument("--warm", nargs="*", default=WARM)
    a.add_argument("--alpha", type=float, default=0.15)
    a.add_argument("--cand-frac", type=float, default=0.5,
                   help="share of the exploration test cases that go to the candidate")
    a.add_argument("--members", type=int, default=16)
    a.add_argument("--interval", type=float, default=60)
    a.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("selftest")
    s.add_argument("--rounds", type=int, default=30)
    i = sub.add_parser("insight")
    i.add_argument("--data", nargs="+", default=WARM)
    i.add_argument("--members", type=int, default=16)
    args = ap.parse_args()
    if args.cmd == "learn":
        learn(args)
    elif args.cmd == "selftest":
        selftest(args)
    else:
        import torch
        torch.set_num_threads(4)
        from tcenv import best_action
        d = load_warm(args.data)
        L = Learner(args.members)
        L.fit(d)
        txt, _ = insight_text(L, d, Data(), to_idx(best_action()), L.scores())
        print(txt)


if __name__ == "__main__":
    main()
