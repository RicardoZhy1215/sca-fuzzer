"""
How similar are the violations a tool finds? Pairwise and nearest-neighbour similarity of violating
programs, per target and tool, against random programs from the same generator config.

Two metrics:
  embed  the "code similarity" of the SpecRL slides: a pretrained sentence encoder
         (sentence-transformers/all-MiniLM-L6-v2, from the local HF cache) embeds the cleaned assembly
         (instrumentation and labels dropped, numbers -> N); cosine similarity. The encoder reads at most
         256 tokens, so a program is embedded in chunks of 16 instructions and the chunk vectors are
         averaged.
  struct Jaccard similarity of the sets of instruction bigrams, an instruction being its mnemonic and
         operand kinds (g = GPR, x = XMM, m = memory, i = immediate).
Sets (the paper's counts: new violations only, RL cut at 10000 programs):
  v4   default64: Revizor 111 (ttk-default64 r0-r2 + extra247), RL GRU 174 / Transformer 159 (A/B run)
  v1   v1.yaml: Revizor 729 (three joined runs), RL GRU 748 (learned modes)
  ssefp normal arm: Revizor 8, RL 178; plus their minimized gadgets
Baseline: 200 random programs per target from Revizor's generator with that config (no policy).

Runs on CPU core 0, one thread (other cores may be measuring time). No executor needed.
  taskset -c 0 python rvzr/SpecRL/genbandit/violation_similarity.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
R = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "figs", "violation-similarity")
MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
                           "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")


# ------------------------------------------------------------------ violation sets
def ab_arm(run, arm, cut=10000):
    tc = [json.loads(l) for l in open(os.path.join(run, "tc.jsonl")) if l.strip()]
    dirs = sorted(glob.glob(os.path.join(run, "violation-*")))
    rows = [r for r in tc if r["stage"] == "violation"]
    assert len(rows) == len(dirs)
    steps = [json.loads(l) for l in open(os.path.join(run, arm, "steps.jsonl"))]
    arm_viol = [(i, s) for i, s in enumerate(steps) if s["stage"] == "violation"]
    mine = [d for d, r in zip(dirs, rows) if r.get("arm") == arm]
    assert len(mine) == len(arm_viol)
    return [d for d, (i, s) in zip(mine, arm_viol) if i < cut and s["new_v"]]


def single(run, cut=10000):
    steps = [json.loads(l) for l in open(os.path.join(run, "steps.jsonl"))]
    dirs = sorted(glob.glob(os.path.join(run, "violation-*")))
    viol = [(i, s) for i, s in enumerate(steps) if s["stage"] == "violation"]
    assert len(viol) == len(dirs), (run, len(viol), len(dirs))
    return [d for d, (i, s) in zip(dirs, viol) if i < cut and s["new_v"]]


def sets():
    T = os.path.join(R, "ttk-default64-261107")
    S1 = os.path.join(HERE, "spectre_v1", "runs")
    SF = os.path.join(HERE, "sse_fp", "runs")
    v4_rev = sum((sorted(glob.glob(f"{T}/r{k}-revizor_default64/violation-*")) for k in range(3)), []) + \
        sorted(glob.glob(f"{R}/revizor-default64-extra247-261110-1555/violation-*"))
    v1_rev = sum((sorted(glob.glob(f"{S1}/{r}/violation-*")) for r in
                  ("revizor-v1yaml-3k-261109", "revizor-v1yaml-3k-part2-n1176-i50-261109-2156",
                   "revizor-v1yaml-10k-part3-n7001-i50-261110-0301")), [])
    ab = os.path.join(R, "ctx-ab-gru-tf-default64-261107")
    sf_rl = single(os.path.join(SF, "ctx-ssefp-normal-20k-261110-2112"), cut=20000)
    sf_rev = sorted(glob.glob(f"{SF}/revizor-ssefp-normal-20k-261110-2112/violation-*"))
    return {
        "v4": {"Revizor": v4_rev, "RL (GRU)": ab_arm(ab, "gru"), "RL (Transformer)": ab_arm(ab, "tf")},
        "v1": {"Revizor": v1_rev, "RL (GRU)": single(os.path.join(S1, "ctx-learn-v1yaml-10k-261110-0014"))},
        "ssefp": {"Revizor": sf_rev, "RL (GRU)": sf_rl},
        "ssefp-min": {"Revizor": sf_rev, "RL (GRU)": sf_rl},
    }


# ------------------------------------------------------------------ program text
def body(path):
    out, inside = [], False
    for l in open(path):
        s = l.split("#")[0].strip()
        if l.strip().startswith(".macro.measurement_start"):
            inside = True
            continue
        if l.strip().startswith(".macro.measurement_end"):
            break
        if inside and s and not s.startswith(".") and "instrumentation" not in l and not s.startswith("jmp"):
            out.append(re.sub(r"\s+", " ", s.lower()))
    return out


def clean(inst):
    return re.sub(r"-?\b(0x[0-9a-f]+|0b[01]+|\d+)\b", "N", inst)


GPR = re.compile(r"^(r[a-d]x|r[sd]i|r[sd]p|r\d+[dwb]?|e[a-d]x|e[sd]i|[a-d][xlh]|[sd]il|[sd]i)$")


def token(inst):
    parts = inst.split(None, 1)
    m = parts[0]
    if m in ("lock", "rep") and len(parts) > 1:
        sub = parts[1].split(None, 1)
        m, parts = m + "_" + sub[0], [m] + sub[1:]
    kinds = []
    if len(parts) > 1:
        for o in parts[1].split(","):
            o = o.strip()
            kinds.append("m" if "ptr" in o or "[" in o else "x" if o.startswith("xmm") else
                         "g" if GPR.match(o) else "i")
    return m + "/" + "".join(kinds)


# ------------------------------------------------------------------ metrics
_tok = _enc = None


def embed(programs):
    global _tok, _enc
    import torch
    torch.set_num_threads(1)
    if _enc is None:
        from transformers import AutoModel, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _enc = AutoModel.from_pretrained(MODEL).eval()
    out = []
    with torch.no_grad():
        for prog in programs:
            chunks = ["\n".join(clean(i) for i in prog[k:k + 16]) for k in range(0, max(len(prog), 1), 16)]
            b = _tok(chunks, padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = _enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)                       # mean pooling, as sentence-transformers
            v = torch.nn.functional.normalize(v, dim=1).mean(0)
            out.append(torch.nn.functional.normalize(v, dim=0).numpy())
    return np.stack(out)


def bigrams(prog):
    t = [token(i) for i in prog]
    return set(zip(t, t[1:])) | {(x,) for x in t}


def jaccard_matrix(A, B):
    return np.array([[len(a & b) / max(len(a | b), 1) for b in B] for a in A])


def stats(M, same):
    """pairwise (off-diagonal if same set) and nearest-neighbour similarity"""
    if same:
        n = len(M)
        iu = np.triu_indices(n, 1)
        pair = M[iu]
        M = M.copy()
        np.fill_diagonal(M, -np.inf)
    else:
        pair = M.ravel()
    nn = M.max(1)
    return pair, nn


# ------------------------------------------------------------------ baseline programs
def baseline(target, n=200):
    sys.path.insert(0, HERE)
    import progrl
    if target == "v4":
        act, sp = json.load(open(os.path.join(R, "ctx-ab-gru-tf-default64-261107", "header.json")))["action"], None
    elif target == "v1":
        sys.path.insert(0, os.path.join(HERE, "spectre_v1"))
        import space_v1ctx as sp
        act = {**sp.CONTROL, "program_size": 32, "n_bb": 2}
    else:
        sys.path.insert(0, os.path.join(HERE, "sse_fp"))
        import space_ssefp as sp
        act = dict(sp.CONTROL)
    cwd = os.getcwd()
    gen, *_ = progrl._offline_generator(act, space=sp)
    os.chdir(cwd)
    path = os.path.join(OUT + "-tmp.asm")
    progs = []
    for _ in range(n):
        gen.create_test_case(path)
        progs.append(body(path))
    os.remove(path)
    return progs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    S = sets()
    rows, curves = [], {}
    for target, groups in S.items():
        progs = {}
        for name, dirs in groups.items():
            f = "minimize/min.asm" if target == "ssefp-min" else "program.asm"
            progs[name] = [body(os.path.join(d, f)) for d in dirs if os.path.exists(os.path.join(d, f))]
        if target != "ssefp-min":
            progs["random programs"] = baseline(target.split("-")[0])
        E = {k: embed(v) for k, v in progs.items()}
        G = {k: [bigrams(p) for p in v] for k, v in progs.items()}
        for name in progs:
            for metric, X in (("embed", E), ("struct", G)):
                M = X[name] @ X[name].T if metric == "embed" else jaccard_matrix(X[name], X[name])
                pair, nn = stats(M, True)
                rows.append((target, name, metric, len(progs[name]), float(np.mean(pair)), float(np.median(pair)),
                             float(np.mean(nn)), float(np.median(nn)), float(np.mean(nn >= 0.95)) if metric == "embed"
                             else float(np.mean(nn >= 0.5))))
                curves[(target, name, metric)] = np.sort(nn)
        # cross: RL vs Revizor nearest neighbour
        for name in progs:
            if name.startswith("RL"):
                for metric, X in (("embed", E), ("struct", G)):
                    M = X[name] @ X["Revizor"].T if metric == "embed" else jaccard_matrix(X[name], X["Revizor"])
                    _, nn = stats(M, False)
                    rows.append((target, f"{name} -> Revizor", metric, len(progs[name]), float(np.mean(M)),
                                 float(np.median(M)), float(np.mean(nn)), float(np.median(nn)), float("nan")))
        print(target, "done", {k: len(v) for k, v in progs.items()}, flush=True)

    with open(OUT + ".csv", "w") as f:
        f.write("target,set,metric,n,pair_mean,pair_median,nn_mean,nn_median,nn_high_share\n")
        for r in rows:
            f.write(",".join(str(x) if not isinstance(x, float) else f"{x:.4f}" for x in r) + "\n")
    with open(OUT + ".txt", "w") as f:
        f.write("nn = similarity to the nearest other program of the same set; nn_high_share = share with "
                "nn >= 0.95 (embed) / >= 0.5 (struct)\n")
        for r in rows:
            f.write(f"{r[0]:10s} {r[1]:28s} {r[2]:6s} n={r[3]:4d}  pair mean {r[4]:.3f} med {r[5]:.3f} | "
                    f"nn mean {r[6]:.3f} med {r[7]:.3f} | high {r[8]:.2f}\n")

    # figure: nearest-neighbour similarity ECDF, per target (columns) and metric (rows)
    TEXT, TEXT2 = "#0b0b0b", "#52514e"
    COL = {"Revizor": "#2a78d6", "RL (GRU)": "#eb6834", "RL (Transformer)": "#1baf7a", "random programs": "#9a9893"}
    targets = ["v4", "v1", "ssefp", "ssefp-min"]
    titles = {"v4": "Spectre v4", "v1": "Spectre v1", "ssefp": "SSE FP", "ssefp-min": "SSE FP (minimized)"}
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4), dpi=200, sharey=True)
    fig.patch.set_facecolor("white")
    for c, t in enumerate(targets):
        for r, metric in enumerate(("embed", "struct")):
            ax = axes[r, c]
            for name in COL:
                y = curves.get((t, name, metric))
                if y is None or len(y) < 2:
                    continue
                ax.plot(y, np.arange(1, len(y) + 1) / len(y), color=COL[name], lw=2,
                        ls=(0, (3, 2)) if name == "random programs" else "-", label=name)
            ax.set_facecolor("white")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(TEXT2)
            ax.tick_params(colors=TEXT2, length=3)
            if r == 0:
                ax.set_title(titles[t], color=TEXT, fontsize=10, loc="left")
            ax.set_xlabel("nearest-neighbour " + ("cosine (embedding)" if metric == "embed" else "Jaccard (bigrams)"),
                          color=TEXT, fontsize=8)
            if c == 0:
                ax.set_ylabel("share of programs", color=TEXT, fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Similarity of violating programs to their nearest neighbour", x=0.02, ha="left",
                 color=TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT + ".png")
    fig.savefig(OUT + ".pdf")
    print(open(OUT + ".txt").read())


if __name__ == "__main__":
    main()
