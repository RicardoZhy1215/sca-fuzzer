"""
Step 2 go/no-go: can a test case's outcome be predicted from what is known before it runs?

Reads tcenv.py's per-test-case log (tc.jsonl + header.json) and reports 5-fold cross-validated
ROC AUC / average precision for "violation" (and the denser "passed the observation filter"):
  config        - logistic regression on the one-hot action (what the bandit already exploits)
  config+feats  - logistic regression and an MLP on the action plus the static program features
  feats | base  - test cases that ran the unperturbed base action only: same config, so any
                  signal comes from the program itself (what a per-test-case decision can use)
Then simulates the skip action with the out-of-fold "feats | base" scores: skip the lowest-scored
fraction q of programs (paying only their pre-execution time) and report violations/hour
relative to testing everything.

  python rvzr/SpecRL/genbandit/analyze_tc.py rvzr/SpecRL/genbandit/runs/tc-collect-<name>
"""
import json
import math
import os
import sys

import numpy as np
import torch

FILTERED = ("spec_filter", "observ_filter")


def load(run):
    h = json.load(open(os.path.join(run, "header.json")))
    R = []
    for line in open(os.path.join(run, "tc.jsonl")):
        try:
            R.append(json.loads(line))
        except json.JSONDecodeError:      # the last line of a log that is still being written
            pass
    sizes = [len(v) for v in h["knobs"].values()]
    off = np.cumsum([0] + sizes[:-1])
    A = np.zeros((len(R), sum(sizes)), np.float32)
    for i, r in enumerate(R):
        A[i, off + np.array(r["a"])] = 1
    X = np.log1p(np.maximum(np.array([r["x"] for r in R], np.float32), 0))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    d = {"A": A, "X": X, "R": R,
         "v": np.array([r["stage"] == "violation" for r in R], np.float32),
         "pass_of": np.array([r["stage"] not in FILTERED for r in R], np.float32),
         "base": np.array([r["pk"] is None for r in R]),
         "s": np.array([r["s"] for r in R]), "pre": np.array([r["pre_ms"] / 1000 for r in R])}
    return h, d


def auc(y, p):
    """ROC AUC via the rank statistic (ties get average ranks)."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ps = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def avg_precision(y, p):
    o = np.argsort(-p, kind="mergesort")
    hits = np.cumsum(y[o])
    prec = hits / np.arange(1, len(y) + 1)
    return float((prec * y[o]).sum() / max(y.sum(), 1))


def fit_predict(Xtr, ytr, Xte, hidden=0, epochs=300, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    n_in = Xtr.shape[1]
    net = torch.nn.Linear(n_in, 1) if not hidden else torch.nn.Sequential(
        torch.nn.Linear(n_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    pos = max(float(ytr.sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(ytr) - pos) / pos))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2 if not hidden else 1e-3, weight_decay=wd)
    xt, yt = torch.tensor(Xtr), torch.tensor(ytr)[:, None]
    bs = len(xt) if not hidden else 1024
    for _ in range(epochs if not hidden else 30):
        perm = torch.randperm(len(xt))
        for k in range(0, len(xt), bs):
            idx = perm[k:k + bs]
            loss = loss_fn(net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(torch.tensor(Xte))[:, 0].numpy()


def cv_scores(X, y, folds=5, seed=0, **kw):
    """Out-of-fold scores, folds stratified by y."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), int)
    for cls in (0, 1):
        idx = np.nonzero(y == cls)[0]
        fold[idx] = rng.permutation(len(idx)) % folds
    out = np.empty(len(y))
    for f in range(folds):
        tr, te = fold != f, fold == f
        out[te] = fit_predict(X[tr], y[tr], X[te], **kw)
    return out


def report(name, y, p):
    n1 = int(y.sum())
    se = math.sqrt(0.25 / max(n1, 1))  # rough upper bound on the AUC's standard error
    print(f"  {name:34s} AUC {auc(y, p):.3f} (±{se:.3f})  AP {avg_precision(y, p):.4f} "
          f"(base rate {y.mean():.4f}, n+ = {n1})")


def skip_curve(y, s, pre, p):
    order = np.argsort(p)                    # lowest scores are skipped first
    v0, t0 = y.sum(), s.sum()
    print("  skip fraction  kept V  time   V/h vs test-all")
    best = (1.0, 0.0)
    for q in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        k = int(q * len(y))
        skip = order[:k]
        keep = np.ones(len(y), bool)
        keep[skip] = False
        v = y[keep].sum()
        t = s[keep].sum() + pre[skip].sum()
        ratio = (v / t) / (v0 / t0)
        best = max(best, (ratio, q))
        print(f"  {q:13.1f}  {v / v0:6.2f}  {t / t0:5.2f}  {ratio:6.3f}")
    print(f"  best: x{best[0]:.3f} at skip fraction {best[1]}")


def main():
    h, d = load(sys.argv[1])
    A, X, v, po, base = d["A"], d["X"], d["v"], d["pass_of"], d["base"]
    AX = np.concatenate([A, X], 1)
    print(f"{len(v)} test cases, {int(v.sum())} violations, {int(base.sum())} unperturbed")
    for label, y in (("violation", v), ("passed observation filter", po)):
        print(f"[{label}]")
        report("config (logistic)", y, cv_scores(A, y))
        report("config+feats (logistic)", y, cv_scores(AX, y))
        report("config+feats (MLP 2x64)", y, cv_scores(AX, y, hidden=64))
        report("feats | base (logistic)", y[base], cv_scores(X[base], y[base]))
        report("feats | base (MLP 2x64)", y[base], cv_scores(X[base], y[base], hidden=64))
    print("[skip action, feats | base logistic scores]")
    p = cv_scores(X[base], v[base])
    skip_curve(v[base], d["s"][base], d["pre"][base], p)


if __name__ == "__main__":
    main()
