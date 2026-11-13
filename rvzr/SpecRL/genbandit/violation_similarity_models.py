"""
Similarity distribution of violating programs under several code encoders (violation_similarity.py
used only MiniLM, a natural-language sentence encoder whose cosines all sit near 0.97).

Encoders (cosine of unit-norm program vectors):
  minilm     sentence-transformers/all-MiniLM-L6-v2 (as in violation_similarity.py: numbers -> N, chunks
             of 16 instructions, chunk vectors averaged)
  clap       CLAP-asm (hustcw/clap-asm): an x86-64 assembly encoder trained contrastively on 195M
             (assembly, description) pairs; the program is one "function", one entry per instruction
  unixcoder  microsoft/unixcoder-base: a code encoder trained with contrastive code-to-code search;
             encoder-only mode, instructions joined by newlines, mean pooling
plus the bigram Jaccard of violation_similarity.py (struct).

Sets: violation_similarity.sets() (the paper's counts), and random programs from Revizor's generator
with the same config (300 for v4 / ssefp, 800 for v1; cached in runs/similarity-models/programs.json so
every encoder sees the same programs).

Reported per target x set x encoder:
  pairwise   distribution of the similarity of all pairs inside a set (does not depend on set size)
  nn         similarity to the nearest other program of the set, on equal-size random subsamples
             (n = the smallest set of the target, 50 draws; the nearest neighbour of a larger set is closer
             by construction)
  dup        share of programs whose nn is above the 95th percentile of the random programs' nn
             (same n): a threshold that is calibrated per encoder
Validation (does the encoder see the leak?): on the 186 SSE-FP violations whose leak mechanism is known
(sse_fp/analysis/normal-20k/classes.tsv: suppression-gated / two-level / direct), AUC that a pair with
the same mechanism is more similar than a pair with different mechanisms, on the full programs and on
the minimized gadgets. 0.5 = the encoder does not see the mechanism.

  cd ~/sca-fuzzer && python rvzr/SpecRL/genbandit/violation_similarity_models.py   (GPU if free; no executor)
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import violation_similarity as vs  # noqa: E402

OUTD = os.path.join(HERE, "runs", "similarity-models")
FIG = os.path.join(HERE, "figs", "violation-similarity-models")
ENCODERS = ["minilm", "clap", "unixcoder"]
LABEL = {"minilm": "MiniLM (text)", "clap": "CLAP-asm (assembly)", "unixcoder": "UniXcoder (code)",
         "struct": "bigram Jaccard"}
TARGETS = ["v4", "v1", "ssefp", "ssefp-min"]
N_RANDOM = {"v4": 300, "v1": 800, "ssefp": 300}
CLASSES = os.path.join(HERE, "sse_fp", "analysis", "normal-20k", "classes.tsv")
MECH = {"gated": "gated", "other": "gated", "two-level": "two-level", "direct": "direct"}   # "other" = the 8
# gates gate_scan's dead test missed, classified as gated by hand (sse_fp/analysis/normal-20k/REPORT.md)


# ------------------------------------------------------------------ programs (cached)
def programs():
    path = os.path.join(OUTD, "programs.json")
    if os.path.exists(path):
        return json.load(open(path))
    S = vs.sets()
    os.makedirs(OUTD, exist_ok=True)
    P = {}
    for target, groups in S.items():
        P[target] = {}
        for name, dirs in groups.items():
            f = "minimize/min.asm" if target == "ssefp-min" else "program.asm"
            keep = [d for d in dirs if os.path.exists(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f))]
            P[target][name] = {"ids": [os.path.relpath(d, HERE) for d in keep],
                               "progs": [vs.body(os.path.join(d, f)) for d in keep]}
        if target != "ssefp-min":
            vs.OUT = os.path.join(OUTD, "baseline")
            r = vs.baseline(target, N_RANDOM[target])
            P[target]["random programs"] = {"ids": [f"random-{i}" for i in range(len(r))], "progs": r}
        print(target, {k: len(v["progs"]) for k, v in P[target].items()}, flush=True)
    json.dump(P, open(path, "w"))
    return P


# ------------------------------------------------------------------ encoders
def enc_minilm():
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, m = AutoTokenizer.from_pretrained(vs.MODEL), AutoModel.from_pretrained(vs.MODEL).to(dev).eval()

    @torch.no_grad()
    def f(progs):
        out = []
        for prog in progs:
            chunks = ["\n".join(vs.clean(i) for i in prog[k:k + 16]) for k in range(0, max(len(prog), 1), 16)]
            b = tok(chunks, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            h = m(**b).last_hidden_state
            a = b["attention_mask"].unsqueeze(-1).float()
            v = torch.nn.functional.normalize((h * a).sum(1) / a.sum(1), dim=1).mean(0)
            out.append(torch.nn.functional.normalize(v, dim=0).cpu().numpy())
        return np.stack(out)
    return f


def enc_clap():
    import clap_asm
    e = clap_asm.ClapAsm()
    return lambda progs: e.embed(progs)


TRUNC = {}


def enc_unixcoder():
    import torch
    from transformers import AutoModel, AutoTokenizer
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--unixcoder-base/snapshots/*"))[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, m = AutoTokenizer.from_pretrained(snap), AutoModel.from_pretrained(snap).to(dev).eval()
    L = m.config.max_position_embeddings - 2 - 4          # RoBERTa positions start at 2; 4 special tokens

    @torch.no_grad()
    def f(progs, batch=16):
        out, cut = [], 0
        for k in range(0, len(progs), batch):
            ids = []
            for p in progs[k:k + batch]:
                t = tok.tokenize("\n".join(p))
                cut += len(t) > L
                ids.append(tok.convert_tokens_to_ids([tok.cls_token, "<encoder-only>", tok.sep_token] + t[:L] +
                                                     [tok.sep_token]))
            n = max(map(len, ids))
            x = torch.tensor([i + [tok.pad_token_id] * (n - len(i)) for i in ids], device=dev)
            a = x.ne(tok.pad_token_id)
            h = m(x, attention_mask=a).last_hidden_state
            v = (h * a.unsqueeze(-1)).sum(1) / a.sum(-1, keepdim=True)
            out.append(torch.nn.functional.normalize(v, dim=1).float().cpu().numpy())
        TRUNC["unixcoder"] = TRUNC.get("unixcoder", 0) + cut
        return np.concatenate(out)
    return f


def embeddings(P):
    E = {}
    for enc in ENCODERS:
        path = os.path.join(OUTD, f"emb-{enc}.npz")
        if os.path.exists(path):
            z = np.load(path)
            E[enc] = {t: {s: z[f"{t}|{s}"] for s in P[t]} for t in P}
            continue
        f = globals()["enc_" + enc]()
        E[enc] = {t: {s: f(v["progs"]) for s, v in P[t].items()} for t in P}
        np.savez(path, **{f"{t}|{s}": x for t in E[enc] for s, x in E[enc][t].items()})
        print("embedded", enc, "(truncated programs:", TRUNC.get(enc, 0), ")", flush=True)
    return E


# ------------------------------------------------------------------ metrics
def sim(enc, A, B):
    return vs.jaccard_matrix(A, B) if enc == "struct" else A @ B.T


def pairwise(enc, X):
    M = sim(enc, X, X)
    return M[np.triu_indices(len(M), 1)]


def nn_sub(enc, X, n, draws=50, seed=0):
    """nearest-neighbour similarity on random subsamples of size n (all of them if n = len)"""
    rng = np.random.default_rng(seed)
    M = sim(enc, X, X).astype(float)
    np.fill_diagonal(M, -np.inf)
    out = []
    for _ in range(draws if n < len(X) else 1):
        idx = rng.choice(len(X), n, replace=False) if n < len(X) else np.arange(len(X))
        out.append(M[np.ix_(idx, idx)].max(1))
    return np.concatenate(out)


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), by ranks"""
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def mechanisms():
    lab = {}
    for l in open(CLASSES):
        f = l.rstrip("\n").split("\t")
        if f[0] == "run":
            continue
        c = MECH.get(f[2].split("(")[0])
        if c:
            lab[f"sse_fp/runs/{f[0]}/{f[1]}"] = c
    return lab


def mech_auc(enc, P, E, target):
    lab = mechanisms()
    ids, X = [], []
    for s in ("Revizor", "RL (GRU)"):
        for i, (key, x) in enumerate(zip(P[target][s]["ids"], E[enc][target][s] if enc != "struct" else
                                         P[target][s]["bigrams"])):
            if key in lab:
                ids.append(lab[key])
                X.append(x)
    X = np.stack(X) if enc != "struct" else X
    M = sim(enc, X, X)
    iu = np.triu_indices(len(ids), 1)
    same = np.array([ids[a] == ids[b] for a, b in zip(*iu)])
    v = M[iu]
    return auc(v[same], v[~same]), len(ids), {c: ids.count(c) for c in sorted(set(ids))}


def main():
    P = programs()
    for t in P:
        for s in P[t]:
            P[t][s]["bigrams"] = [vs.bigrams(p) for p in P[t][s]["progs"]]
    E = embeddings(P)
    rows, dist = [], {}
    lines = ["pairwise / nn: median [IQR]; nn on equal-size subsamples (n shown); dup = share of nn above the "
             "95th percentile of random programs' nn (same n)", ""]
    for enc in ENCODERS + ["struct"]:
        lines.append(f"== {LABEL[enc]}")
        for t in TARGETS:
            sets = list(P[t])
            n = min(len(P[t][s]["progs"]) for s in sets if len(P[t][s]["progs"]) >= 20)
            X = {s: (P[t][s]["bigrams"] if enc == "struct" else E[enc][t][s]) for s in sets}
            ref = nn_sub(enc, X["random programs"], n) if "random programs" in X else None
            thr = np.quantile(ref, 0.95) if ref is not None else None
            for s in sets:
                pw = pairwise(enc, X[s])
                nn = nn_sub(enc, X[s], min(n, len(X[s])))
                dup = float(np.mean(nn > thr)) if thr is not None else float("nan")
                dist[(enc, t, s)] = (pw, nn)
                q = lambda a: f"{np.median(a):.3f} [{np.quantile(a, .25):.3f}-{np.quantile(a, .75):.3f}]"  # noqa: E731
                lines.append(f"  {t:9s} {s:16s} N={len(X[s]):4d} pairwise {q(pw)}  nn(n={min(n, len(X[s]))}) {q(nn)}"
                             f"  dup {dup:.2f}")
                rows.append((enc, t, s, len(X[s]), np.median(pw), np.mean(pw), np.median(nn), np.mean(nn), dup))
            for s in sets:                                # cross: nearest Revizor violation
                if s.startswith("RL") and "Revizor" in X and len(X["Revizor"]) >= 20:
                    M = sim(enc, X[s], X["Revizor"])
                    lines.append(f"  {t:9s} {s + ' -> Revizor':28s} nn {np.median(M.max(1)):.3f}"
                                 + (f"   (random -> Revizor nn {np.median(sim(enc, X['random programs'], X['Revizor']).max(1)):.3f})"
                                    if "random programs" in X else ""))
        for t in ("ssefp", "ssefp-min"):
            a, k, c = mech_auc(enc, P, E, t)
            lines.append(f"  mechanism AUC on {t:9s} {a:.3f}  ({k} violations: {c})")
            rows.append((enc, t, "mechanism AUC", k, a, float("nan"), float("nan"), float("nan"), float("nan")))
        lines.append("")
    if TRUNC:
        lines.append(f"unixcoder: {TRUNC['unixcoder']} programs longer than its 1020-token window (truncated)")
    open(FIG + ".txt", "w").write("\n".join(lines) + "\n")
    with open(FIG + ".csv", "w") as f:
        f.write("encoder,target,set,n,pairwise_median,pairwise_mean,nn_median,nn_mean,dup_share\n")
        for r in rows:
            f.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print("\n".join(lines))
    plot(dist, P)


def plot(dist, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    TEXT, TEXT2 = "#0b0b0b", "#52514e"
    COL = {"Revizor": "#2a78d6", "RL (GRU)": "#eb6834", "RL (Transformer)": "#1baf7a", "random programs": "#9a9893"}
    titles = {"v4": "Spectre v4", "v1": "Spectre v1", "ssefp": "SSE FP", "ssefp-min": "SSE FP (minimized)"}
    encs = ENCODERS + ["struct"]
    for which, k in (("pairwise", 0), ("nn", 1)):
        fig, axes = plt.subplots(len(encs), len(TARGETS), figsize=(12, 2.3 * len(encs)), dpi=200)
        fig.patch.set_facecolor("white")
        for r, enc in enumerate(encs):
            lo = min(np.quantile(dist[(enc, t, s)][k], 0.005) for t in TARGETS for s in P[t]
                     if len(dist[(enc, t, s)][k]) > 1)
            hi = max(dist[(enc, t, s)][k].max() for t in TARGETS for s in P[t] if len(dist[(enc, t, s)][k]) > 1)
            bins = np.linspace(lo, hi, 50)
            for c, t in enumerate(TARGETS):
                ax = axes[r, c]
                for s in COL:
                    if (enc, t, s) not in dist or len(dist[(enc, t, s)][k]) < 10:
                        continue
                    ax.hist(dist[(enc, t, s)][k], bins=bins, density=True, histtype="step", lw=1.6, color=COL[s],
                            ls=(0, (3, 2)) if s == "random programs" else "-", label=s)
                ax.set_facecolor("white")
                ax.set_yticks([])
                for sp in ("top", "right", "left"):
                    ax.spines[sp].set_visible(False)
                ax.spines["bottom"].set_color(TEXT2)
                ax.tick_params(colors=TEXT2, length=3, labelsize=7)
                if r == 0:
                    ax.set_title(titles[t], color=TEXT, fontsize=10, loc="left")
                if c == 0:
                    ax.set_ylabel(LABEL[enc], color=TEXT, fontsize=8.5)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
        fig.supxlabel(("pairwise similarity inside a set" if which == "pairwise" else
                       "similarity to the nearest other program of the set (equal-size subsamples)"),
                      color=TEXT, fontsize=9, y=0.035)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(f"{FIG}-{which}.png")
        fig.savefig(f"{FIG}-{which}.pdf")
        print(f"{FIG}-{which}.png")


if __name__ == "__main__":
    main()
