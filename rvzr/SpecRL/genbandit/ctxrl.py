"""
Context-conditioned program RL: the hardware observation of the previous program guides the
generation of the next one.

Episode = N consecutive programs (test cases). Step k sees a context made of program k-1's
hardware observation (stage reached, dispatched - retired ops per input, inputs whose fenced
trace differs, whether it produced a NEW violation) and program k-1 itself as the reference.
The policy first picks a mode
    FRESH - write a new program (the progrl decision sequence: instruction, register families)
    RERUN - run the reference again unchanged (only the inputs are new)
    EDIT  - for every position: COPY the reference instruction / keep its instruction but redraw
            its operands (NEW_OPS) / write a new instruction (NEW_INS)
then makes the decisions of that mode, conditioned (GRU initial state) on the context and, per
position, on the reference instruction. One hardware run per program, as in progrl.

Reward: 1 for a violation by a program that has not been rewarded before (hash of the full
program, including the origin tags that fix its immediates) -- re-running a known violating
program earns nothing, any changed program (a variant) earns again.

Learning: PPO-clip, token level, with GAE advantages (gamma 1, lambda 0.9) from a critic V(context)
(the edit pilot showed the previous program's observation predicts the next reward: AUC 0.76),
self-imitation on the rewarded programs, KL to the prior (Revizor for FRESH; mode prior
0.9/0.05/0.05; copy prior 1-2/64, 1/64, 1/64).

Several basic blocks (Spectre v1, tcenv.py --space v1): the policy still writes the P
instructions in program order; a new program's block sizes are drawn as Revizor draws them
(multinomial, fresh_layout) and each position's block is one more query embedding. Revizor
draws the CFG and the jumps from the program's structure tag, so a RERUN / EDIT gets the
reference's blocks, CFG and jumps exactly.

Speed: E episodes are interleaved (round robin), so while the hardware runs one episode's
program the agent (cores 4-7) already writes the next program of another episode.

  python rvzr/SpecRL/genbandit/ctxrl.py synthetic [--fresh-only]   # planted world, no hardware
  tcenv.py --policy ctx --ctx-policy {learn,rule,fresh}            # on the hardware
"""
import argparse
import hashlib
import json
import math
import os
import queue
import random
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from progrl import K_SPEC, K_REG, K_MEM, NEG, build_table, install  # noqa: E402

K_MODE, K_COPY = 4, 5
FRESH, RERUN, EDIT = 0, 1, 2
COPY, NEW_OPS, NEW_INS = 0, 1, 2
MODES = ["fresh", "rerun", "edit"]
MODE_PRIOR = [0.90, 0.05, 0.05]
N_CTX = 16


def stage_class(stage, new_v):
    if stage == "spec_filter":
        return 0
    if stage == "observ_filter":
        return 1
    if stage == "violation":
        return 3 if new_v else 4
    return 2                                    # near miss: passed the observation filter


def ctx_vector(step, N, prev):
    """prev = None for the first program, else {"stage", "info", "new_v", "mode", "copied"}"""
    v = np.zeros(N_CTX, np.float32)
    v[1] = step / N
    if prev is None:
        v[0] = 1.0
        return v
    info = prev["info"] or {}
    v[2 + stage_class(prev["stage"], prev["new_v"])] = 1.0          # 2..6
    v[7] = np.clip(info.get("ex_max", 0) / 100, -1, 3)
    v[8] = np.clip(info.get("ex_mean", 0) / 50, -1, 3)
    n = max(info.get("n_in", 1), 1)
    v[9] = info.get("ex_pos", 0) / n
    v[10] = (info.get("fdiv") or 0) / n
    v[11 + prev["mode"]] = 1.0                                       # 11..13
    v[14] = prev["copied"]
    v[15] = 1.0 if "fdiv" in info else 0.0
    return v


def prog_hash(prog):
    return hashlib.sha1(json.dumps(prog).encode()).hexdigest()


# ============================================================================ policy
TF_MAX_T = 512        # tokens per program incl. the context token (edit worst case ~258)


def make_policy(table, hidden=128, emb=64, seed=0, copy_new=2 / 64, arch="gru", tf_layers=2,
                tf_heads=4, tf_ff=512):
    """arch "gru": the context enters once, as the GRU's initial state; each token's input is the
    sum of its embeddings (previous token value + query). arch "tf": a causal pre-LN Transformer
    over [context token, token inputs], width `hidden` (embeddings are `hidden` wide too), so every
    decision can attend to the context and to any earlier decision directly. Both expose
    start(ctx) / step(x, state, rows) (incremental sampling) and seq(x, ctx) (teacher forcing)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    n_specs, P, n_bb = len(table["names"]), table["program_size"], table["n_bb"]
    assert arch in ("gru", "tf")
    if arch == "tf":
        emb = hidden
    sizes = {K_SPEC: n_specs, K_REG: 6, K_MEM: 6, K_MODE: 3, K_COPY: 3}

    class Block(nn.Module):
        def __init__(self, d, nh, ff):
            super().__init__()
            self.nh = nh
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

        def _heads(self, t):                                   # (B, T, d) -> (B, nh, T, d/nh)
            B, T, d = t.shape
            return t.view(B, T, self.nh, d // self.nh).transpose(1, 2)

        def forward(self, x):                                  # (B, T, d), causal
            B, T, d = x.shape
            q, k, v = (self._heads(t) for t in self.qkv(self.ln1(x)).chunk(3, -1))
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
            return x + self.mlp(self.ln2(x))

        def step(self, x, kv):                                 # x (B, d); kv: cached keys/values
            B, d = x.shape
            q, k, v = (self._heads(t[:, None]) for t in self.qkv(self.ln1(x)).chunk(3, -1))
            if kv is not None:
                k, v = torch.cat([kv[0], k], 2), torch.cat([kv[1], v], 2)
            a = F.scaled_dot_product_attention(q, k, v)        # the new token sees all cached ones
            x = x + self.proj(a.transpose(1, 2).reshape(B, d))
            return x + self.mlp(self.ln2(x)), (k, v)

    class CtxPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = arch
            o = 1
            self.off = {}
            for k in (K_SPEC, K_REG, K_MEM, K_MODE, K_COPY):
                self.off[k] = o
                o += sizes[k]
            self.val = nn.Embedding(o, emb)                    # 0 = BOS
            self.q_kind = nn.Embedding(6, emb)
            self.q_pos = nn.Embedding(4, emb)
            self.q_sd = nn.Embedding(4, emb)
            self.q_spec = nn.Embedding(n_specs + 1, emb)       # current instruction (n_specs: none)
            self.q_ipos = nn.Embedding(P + 1, emb)             # position (P: not per position)
            self.q_ref = nn.Embedding(n_specs + 1, emb)        # reference instruction at position
            # basic block of the position (n_bb: not per position); only with several blocks, so
            # single-block policies keep their parameter set
            self.q_bb = nn.Embedding(n_bb + 1, emb) if n_bb > 1 else None
            if arch == "gru":
                self.ctx = nn.Sequential(nn.Linear(N_CTX, hidden), nn.Tanh())
                self.gru = nn.GRU(emb, hidden, batch_first=True)
            else:
                self.ctx_tok = nn.Linear(N_CTX, hidden)        # the context is token 0
                self.tpos = nn.Embedding(TF_MAX_T, hidden)     # token index in the sequence
                self.blocks = nn.ModuleList(Block(hidden, tf_heads, tf_ff) for _ in range(tf_layers))
                self.ln_f = nn.LayerNorm(hidden)
            self.heads = nn.ModuleDict({str(k): nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n)) for k, n in sizes.items()})
            for h in self.heads.values():
                nn.init.zeros_(h[-1].weight)
                nn.init.zeros_(h[-1].bias)
            self.critic = nn.Sequential(nn.Linear(N_CTX, 64), nn.Tanh(), nn.Linear(64, 1))
            self.register_buffer("spec_prior", torch.tensor(table["spec_logprior"]))
            self.register_buffer("slot_prior", torch.tensor(table["slot_prior"] or [[0.0] * 6],
                                                            dtype=torch.float32))
            self.register_buffer("mode_prior", torch.log(torch.tensor(MODE_PRIOR)))
            cp = [math.log(1 - copy_new), math.log(copy_new / 2), math.log(copy_new / 2)]
            # row 0: NEW_OPS impossible (the reference instruction has no operand choices)
            self.register_buffer("copy_prior", torch.tensor([[cp[0], NEG, cp[2]], cp]))

        def query(self, kind, pos, sd, spec, ipos, ref, bb=None):
            q = (self.q_kind(kind) + self.q_pos(pos) + self.q_sd(sd) + self.q_spec(spec)
                 + self.q_ipos(ipos) + self.q_ref(ref))
            return q + self.q_bb(bb) if self.q_bb is not None else q

        def prior(self, kind, sid):
            n = len(sid)
            if kind == K_SPEC:
                return self.spec_prior.expand(n, -1)
            if kind in (K_REG, K_MEM):
                return self.slot_prior[sid]
            if kind == K_MODE:
                return self.mode_prior.expand(n, -1)
            return self.copy_prior[sid]

        def cell(self, x, h):
            g = self.gru
            gi = x @ g.weight_ih_l0.T + g.bias_ih_l0
            gh = h @ g.weight_hh_l0.T + g.bias_hh_l0
            ir, iz, i_n = gi.chunk(3, 1)
            hr, hz, hn = gh.chunk(3, 1)
            r = torch.sigmoid(ir + hr)
            z = torch.sigmoid(iz + hz)
            n_ = torch.tanh(i_n + r * hn)
            return (1 - z) * n_ + z * h

        # ---- incremental (sampling): rows = the batch rows still generating, in order
        def start(self, c):
            if self.arch == "gru":
                return {"h": self.ctx(c)}
            x = self.ctx_tok(c) + self.tpos.weight[0]
            kv = []
            for b in self.blocks:
                x, c_ = b.step(x, None)
                kv.append(c_)
            return {"kv": kv, "rows": list(range(len(c))), "t": 1}

        def step(self, x, state, rows):
            if self.arch == "gru":
                ha = self.cell(x, state["h"][rows])
                state["h"][rows] = ha
                return ha
            if rows != state["rows"]:                          # finished rows leave the cache
                idx = torch.tensor([state["rows"].index(r) for r in rows])
                state["kv"] = [(k[idx], v[idx]) for k, v in state["kv"]]
                state["rows"] = list(rows)
            x = x + self.tpos.weight[state["t"]]
            for i, b in enumerate(self.blocks):
                x, state["kv"][i] = b.step(x, state["kv"][i])
            state["t"] += 1
            return self.ln_f(x)

        # ---- whole sequences (teacher forcing): x (B, T, emb), c (B, N_CTX) -> (B, T, hidden)
        def seq(self, x, c):
            if self.arch == "gru":
                h, _ = self.gru(x, self.ctx(c).unsqueeze(0).contiguous())
                return h
            X = torch.cat([self.ctx_tok(c)[:, None], x], 1)
            X = X + self.tpos.weight[:X.shape[1]]
            for b in self.blocks:
                X = b(X)
            return self.ln_f(X)[:, 1:]

    return CtxPolicy()


def fresh_layout(rng, P, n_bb):
    """Basic block of every position of a new program, in program order: Revizor puts each
    instruction into a uniformly chosen block, so the block sizes are multinomial(P, 1/n_bb);
    the blocks are laid out one after the other."""
    if n_bb <= 1:
        return [0] * P
    return [b for b, c in enumerate(rng.multinomial(P, [1.0 / n_bb] * n_bb)) for _ in range(c)]


# token = (kind, value, pos, sd, qspec, sid, logp, prev_id, ipos, ref, forced, bb)
class CtxSampler:
    def __init__(self, table, pol, rng):
        self.t, self.pol, self.rng = table, pol, rng
        self.n_specs = len(table["names"])
        self.P = table["program_size"]
        self.n_bb = table["n_bb"]

    def sample(self, ctxs, refs, forced=None):
        """ctxs: (B, N_CTX); refs: list of reference programs or None; forced: per program None
        (the policy picks the mode) or a mode fixed by a rule -- it is still fed to the GRU as a
        token (same input structure as a learned mode) but flagged and excluded from the loss.
        Returns, per program, (mode, instruction list with tag None for new instructions, tokens,
        copied fraction). Instructions are [spec, register families, basic block, tag]: a new
        program's blocks come from fresh_layout, RERUN / EDIT keep the reference's."""
        forced = forced if forced is not None else [None] * len(refs)
        import torch
        t, pol, P, NS = self.t, self.pol, self.P, self.n_specs
        B = len(refs)
        with torch.no_grad():
            state = pol.start(torch.from_numpy(np.asarray(ctxs, np.float32)))
        st = []
        for b in range(B):
            st.append({"ph": "mode" if refs[b] is not None else "spec", "i": 0, "j": 0,
                       "spec": None, "mode": FRESH if refs[b] is None else None,
                       "prog": [], "toks": [], "prev": 0, "done": False, "copied": 0,
                       "layout": fresh_layout(self.rng, P, self.n_bb) if refs[b] is None else None})
        while True:
            act = [b for b in range(B) if not st[b]["done"]]
            if not act:
                break
            q = np.zeros((len(act), 8), np.int64)        # kind pos sd qspec sid ipos ref bb
            for k, b in enumerate(act):
                s, ref = st[b], refs[b]
                i = s["i"]
                rs = ref[i][0] if ref is not None and i < P else NS
                bb = s["layout"][i] if s["layout"] is not None and i < P else self.n_bb
                if s["ph"] == "mode":
                    q[k] = (K_MODE, 0, 0, NS, 0, P, NS, self.n_bb)
                elif s["ph"] == "copy":
                    q[k] = (K_COPY, 0, 0, NS, 1 if t["slots"][rs] else 0, i, rs, bb)
                elif s["ph"] == "spec":
                    q[k] = (K_SPEC, 0, 0, NS, 0, i, rs, bb)
                else:
                    sl = t["slots"][s["spec"]][s["j"]]
                    q[k] = (sl["kind"], sl["pos"], 2 * sl["src"] + sl["dest"], s["spec"], sl["sid"], i, rs, bb)
            T = torch.from_numpy
            with torch.no_grad():
                prev = T(np.array([st[b]["prev"] for b in act], np.int64))
                x = pol.val(prev) + pol.query(*(T(q[:, c]) for c in (0, 1, 2, 3, 5, 6, 7)))
                ha = pol.step(x, state, act)
                val = np.empty(len(act), np.int64)
                lp = np.empty(len(act), np.float32)
                for kd in np.unique(q[:, 0]):
                    m = np.nonzero(q[:, 0] == kd)[0]
                    lg = pol.prior(int(kd), T(q[m, 4])) + pol.heads[str(int(kd))](ha[m])
                    lsm = torch.log_softmax(lg, 1).numpy()
                    pr = np.exp(lsm)
                    u = self.rng.random((len(m), 1))
                    v = np.minimum((pr.cumsum(1) < u).sum(1), pr.shape[1] - 1)
                    v = np.where(lsm[np.arange(len(m)), v] < -1e8, pr.argmax(1), v)
                    val[m] = v
                    lp[m] = lsm[np.arange(len(m)), v]
            fl = np.zeros(len(act), np.int64)
            for k, b in enumerate(act):
                if q[k, 0] == K_MODE and forced[b] is not None:
                    val[k], lp[k], fl[k] = forced[b], 0.0, 1
            for k, b in enumerate(act):
                s, ref = st[b], refs[b]
                kd, v = int(q[k, 0]), int(val[k])
                s["toks"].append((kd, v, int(q[k, 1]), int(q[k, 2]), int(q[k, 3]), int(q[k, 4]),
                                  float(lp[k]), int(s["prev"]), int(q[k, 5]), int(q[k, 6]), int(fl[k]),
                                  int(q[k, 7])))
                s["prev"] = pol.off[kd] + v
                self._advance(s, ref, kd, v)
        return [(s["mode"], s["prog"], s["toks"], s["copied"] / P) for s in st]

    def _advance(self, s, ref, kd, v):
        t, P = self.t, self.P

        def next_instruction():
            s["i"] += 1
            if s["i"] >= P:
                s["done"] = True
            else:
                s["ph"] = "copy" if s["mode"] == EDIT else "spec"

        def start_slots():
            if t["slots"][s["spec"]]:
                s["ph"], s["j"] = "slot", 0
            else:
                next_instruction()

        if kd == K_MODE:
            s["mode"] = v
            s["layout"] = fresh_layout(self.rng, P, self.n_bb) if v == FRESH else [x[2] for x in ref]
            if v == RERUN:
                s["prog"] = [list(x) for x in ref]
                s["copied"] = P
                s["done"] = True
            else:
                s["ph"] = "copy" if v == EDIT else "spec"
        elif kd == K_COPY:
            r = ref[s["i"]]
            if v == COPY:
                s["prog"].append(list(r))
                s["copied"] += 1
                next_instruction()
            elif v == NEW_OPS:
                s["spec"] = r[0]
                s["prog"].append([r[0], [], s["layout"][s["i"]], None])
                start_slots()
            else:
                s["ph"] = "spec"
        elif kd == K_SPEC:
            s["spec"] = v
            s["prog"].append([v, [], s["layout"][s["i"]], None])
            start_slots()
        else:
            s["prog"][-1][1].append(v)
            s["j"] += 1
            if s["j"] >= len(t["slots"][s["spec"]]):
                next_instruction()


def evaluate(pol, toks_batch, ctxs):
    import torch
    B = len(toks_batch)
    T = max(len(x) for x in toks_batch)
    W = max(len(x[0]) for x in toks_batch if x)
    arr = np.zeros((B, T, W), np.float64)
    valid = np.zeros((B, T), bool)
    for i, x in enumerate(toks_batch):
        arr[i, :len(x)] = np.asarray(x, np.float64)
        valid[i, :len(x)] = True
    a = torch.from_numpy(arr)
    kind, val, pos, sd, qs, sid, prev, ipos, ref = (a[..., c].long() for c in (0, 1, 2, 3, 4, 5, 7, 8, 9))
    bb = a[..., 11].long() if W > 11 else None
    x = pol.val(prev) + pol.query(kind, pos, sd, qs, ipos, ref, bb)
    h = pol.seq(x, torch.from_numpy(np.asarray(ctxs, np.float32)))
    logp, kl, ent = torch.zeros(B, T), torch.zeros(B, T), torch.zeros(B, T)
    if W > 10:
        valid &= arr[..., 10] == 0                 # rule-forced mode tokens: input only
    vt = torch.from_numpy(valid)
    for kd in (K_SPEC, K_REG, K_MEM, K_MODE, K_COPY):
        m = (kind == kd) & vt
        if not m.any():
            continue
        pri = pol.prior(kd, sid[m])
        lg = pri + pol.heads[str(kd)](h[m])
        lsm = torch.log_softmax(lg, 1)
        lsm0 = torch.log_softmax(pri, 1)
        pr = lsm.exp()
        ok = pri > NEG / 2
        logp[m] = lsm.gather(1, val[m].unsqueeze(1)).squeeze(1)
        kl[m] = torch.where(ok, pr * (lsm - lsm0), torch.zeros_like(pr)).sum(1)
        ent[m] = -torch.where(ok, pr * lsm, torch.zeros_like(pr)).sum(1)
    return logp, kl, ent, vt, torch.from_numpy(arr[..., 6]).float(), kind


class CtxLearner:
    def __init__(self, table, seed=0, lr=3e-4, kl=0.1, sil=1.0, vf=0.5, clip=0.2, epochs=2,
                 lam=0.9, sil_batch=64, buf_cap=5000, arch="gru", tf=None):
        import torch
        self.torch = torch
        self.pol = make_policy(table, seed=seed, arch=arch, **(tf or {}))
        self.opt = torch.optim.Adam(self.pol.parameters(), lr=lr)
        self.sampler = CtxSampler(table, self.pol, np.random.default_rng(seed + 1))
        self.kl, self.sil, self.vf, self.clip, self.epochs, self.lam = kl, sil, vf, clip, epochs, lam
        self.sil_batch, self.buf_cap = sil_batch, buf_cap
        self.pos = []
        self.rng = random.Random(seed)
        self.n_upd = 0

    def values(self, ctxs):
        with self.torch.no_grad():
            return self.pol.critic(self.torch.from_numpy(np.asarray(ctxs, np.float32)))[:, 0].numpy()

    def gae(self, steps):
        """steps of one finished episode, in order; adds adv and ret (gamma 1, V after end = 0)."""
        V = self.values([s["ctx"] for s in steps])
        adv, nxt = 0.0, 0.0
        for k in range(len(steps) - 1, -1, -1):
            d = steps[k]["r"] + nxt - V[k]
            adv = d + self.lam * adv
            steps[k]["adv"], steps[k]["ret"] = float(adv), float(adv + V[k])
            nxt = V[k]

    def add_positive(self, step):
        self.pos.append((step["toks"], step["ctx"]))
        if len(self.pos) > self.buf_cap:
            self.pos.pop(0)

    def update(self, steps):
        torch = self.torch
        t0 = time.time()
        A = torch.tensor([s["adv"] for s in steps], dtype=torch.float32)
        A = (A - A.mean()) / (A.std() + 1e-8)
        R = torch.tensor([s["ret"] for s in steps], dtype=torch.float32)
        toks, ctx = [s["toks"] for s in steps], np.stack([s["ctx"] for s in steps])
        cx = torch.from_numpy(ctx)
        for _ in range(self.epochs):
            logp, kl, ent, vt, old, _ = evaluate(self.pol, toks, ctx)
            ratio = torch.exp(logp - old)
            Ab = A[:, None].expand_as(ratio)
            pg = -torch.minimum(ratio * Ab, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * Ab)[vt].mean() \
                if vt.any() else torch.tensor(0.0)
            klm = kl[vt].mean() if vt.any() else torch.tensor(0.0)
            v = self.pol.critic(cx)[:, 0]
            vloss = ((v - R) ** 2).mean()
            loss = pg + self.kl * klm + self.vf * vloss
            sil = torch.tensor(0.0)
            if self.pos and self.sil:
                sb = self.rng.sample(self.pos, min(self.sil_batch, len(self.pos)))
                lps, _, _, vts, _, kds = evaluate(self.pol, [x[0] for x in sb], np.stack([x[1] for x in sb]))
                # not the MODE decision: most positives come from the current majority mode, so
                # imitating their mode would only entrench it; which mode to use in which context is
                # learned from the advantage (policy gradient) alone
                m = vts & (kds != K_MODE)
                sil = -lps[m].mean() if m.any() else torch.tensor(0.0)
                loss = loss + self.sil * sil
            self.opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.pol.parameters(), 1.0)
            self.opt.step()
        self.n_upd += 1
        return {"pg": float(pg), "kl": float(klm), "vloss": float(vloss), "sil": float(sil),
                "ent": float(ent[vt].mean()) if vt.any() else 0.0, "grad": float(gn), "v_mean": float(v.mean()),
                "ret_mean": float(R.mean()), "upd_s": round(time.time() - t0, 2)}


# ============================================================================ non-learning policies
class PriorPrograms:
    """numpy sampler of Revizor's distribution + single random edits (rule / fresh baselines)."""

    def __init__(self, table, rng):
        self.t, self.rng = table, rng
        p = np.exp(np.array(table["spec_logprior"]))
        self.p_spec = p / p.sum()
        self.p_slot = [np.exp(np.array(sp)) / np.exp(np.array(sp)).sum() for sp in table["slot_prior"]]

    def inst(self, spec=None, bb=0):
        s = int(self.rng.choice(len(self.p_spec), p=self.p_spec)) if spec is None else spec
        return [s, [int(self.rng.choice(6, p=self.p_slot[sl["sid"]])) for sl in self.t["slots"][s]], bb, None]

    def fresh(self):
        return [self.inst(bb=b) for b in fresh_layout(self.rng, self.t["program_size"], self.t["n_bb"])]

    def edit(self, ref):
        prog = [list(x) for x in ref]
        pos = int(self.rng.integers(len(prog)))
        if self.rng.random() < 0.5 or not self.t["slots"][prog[pos][0]]:
            old = prog[pos][0]
            new = old
            while new == old:
                new = int(self.rng.choice(len(self.p_spec), p=self.p_spec))
            prog[pos] = self.inst(new, prog[pos][2])
        else:
            prog[pos] = self.inst(prog[pos][0], prog[pos][2])
        return prog


def rule_mode(prev, ex_min=60):
    """Edit-pilot rule: keep working on a promising program, otherwise write a new one."""
    if prev is None:
        return FRESH
    if prev["stage"] == "violation":
        return EDIT if prev["new_v"] else FRESH          # a new violation -> look for variants
    near = prev["stage"] not in ("spec_filter", "observ_filter")
    return RERUN if near and (prev["info"] or {}).get("ex_max", 0) >= ex_min else FRESH


def rule2_mode(prev, rng):
    """After a violation (its lineage is spent) or a filtered program: write a new one. After a
    near miss (passed the observation filter, no violation): rerun or edit it, 50/50."""
    if prev is None or prev["stage"] in ("violation", "spec_filter", "observ_filter"):
        return FRESH
    return RERUN if rng.random() < 0.5 else EDIT


# ============================================================================ agent process
def agent_main(table, out, prog_q, res_q, opts):
    try:
        os.sched_setaffinity(0, set(opts.get("cores", [4, 5, 6, 7])))
    except Exception:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(opts.get("threads", 4))
    E, N, pol_kind = opts.get("E", 16), opts.get("N", 8), opts.get("policy", "learn")
    rng = np.random.default_rng(opts.get("seed", 0))
    learner = CtxLearner(table, seed=opts.get("seed", 0), lr=opts.get("lr", 3e-4),
                         kl=opts.get("kl", 0.1), sil=opts.get("sil", 1.0), arch=opts.get("arch", "gru"),
                         tf=opts.get("tf")) if pol_kind == "learn" else None
    if learner is not None:
        n_par = sum(p.numel() for p in learner.pol.parameters())
        json.dump({"arch": opts.get("arch", "gru"), "tf": opts.get("tf"), "params": n_par,
                   "opts": {k: v for k, v in opts.items() if k != "tf"}},
                  open(os.path.join(out, "agent_header.json"), "w"))
    mode_kind = opts.get("mode", "learn")             # learn: policy's mode head; rule2: rule2_mode
    if learner is not None and opts.get("init"):
        learner.pol.load_state_dict(torch.load(opts["init"]))    # warm start (same instruction set)
    violated = set()                                   # lineages that already produced a violation
    lineage_ctr = 0
    prior = PriorPrograms(table, rng)
    log = open(os.path.join(out, "steps.jsonl"), "a", buffering=1 << 16)
    alog = open(os.path.join(out, "agent.jsonl"), "a")
    rewarded = set()
    slots = [None] * E
    pending = {}
    pid = 0
    ep_counter = 0
    batch = []
    upd_every = opts.get("update_every", 256)
    n_res = n_new = 0
    t0 = time.time()

    def new_episode(e):
        nonlocal ep_counter
        slots[e] = {"ep": ep_counter, "step": 0, "seed": int(rng.integers(1 << 62)), "ref": None,
                    "prev": None, "steps": []}
        ep_counter += 1

    def emit(e):
        nonlocal pid, lineage_ctr
        sl = slots[e]
        ctx = ctx_vector(sl["step"], N, sl["prev"])
        if pol_kind == "learn":
            fm = rule2_mode(sl["prev"], rng) if (mode_kind == "rule2" and sl["ref"] is not None) else None
            mode, prog, toks, copied = learner.sampler.sample(ctx[None], [sl["ref"]], [fm])[0]
        else:
            mode = rule_mode(sl["prev"]) if pol_kind == "rule" else FRESH
            toks = None
            if mode == FRESH:
                prog = prior.fresh()
            elif mode == RERUN:
                prog = [list(x) for x in sl["ref"]]
            else:
                prog = prior.edit(sl["ref"])
            copied = sum(1 for a, b in zip(prog, sl["ref"] or []) if a == b) / len(prog)
        tag = f"{sl['seed']}:{sl['step']}"
        for x in prog:
            if x[3] is None:
                x[3] = tag
        if mode == FRESH or sl.get("lineage") is None:
            lineage_ctr += 1
            lin = lineage_ctr
            struct = tag                               # seeds Revizor's CFG and jumps
        else:
            lin = sl["lineage"]                        # rerun / edit inherit the reference's lineage
            struct = sl["struct"]                      # ... and its CFG and jumps
        pending[pid] = {"slot": e, "ep": sl["ep"], "step": sl["step"], "mode": mode, "prog": prog,
                        "toks": toks, "ctx": ctx, "copied": copied, "lineage": lin, "struct": struct}
        prog_q.put((pid, e, prog, struct))
        pid += 1

    for e in range(E):
        new_episode(e)
        emit(e)
    while True:
        try:
            msg = res_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if msg is None:
            break
        rid, stage, secs, info = msg
        st = pending.pop(rid)
        h = prog_hash(st["prog"])
        new_v_hash = stage == "violation" and h not in rewarded
        # reward: the FIRST violation of a lineage (a program and everything rerun / edited from it)
        new_v = new_v_hash and st["lineage"] not in violated
        if stage == "violation":
            violated.add(st["lineage"])
        if new_v_hash:
            rewarded.add(h)
        if new_v:
            n_new += 1
        n_res += 1
        st["r"] = 1.0 if new_v else 0.0
        e = st["slot"]
        sl = slots[e]
        sl["steps"].append(st)
        log.write(json.dumps({"ep": st["ep"], "step": st["step"], "slot": e, "mode": MODES[st["mode"]],
                              "copied": round(st["copied"], 3), "stage": stage, "s": round(secs, 4),
                              "new_v": new_v, "dup_v": stage == "violation" and not new_v,
                              "new_v_hash": new_v_hash, "lineage": st["lineage"],
                              **(info or {})}) + "\n")
        if learner is not None and new_v:
            learner.add_positive(st)
        sl["prev"] = {"stage": stage, "info": info, "new_v": new_v, "mode": st["mode"], "copied": st["copied"]}
        sl["ref"] = st["prog"]
        sl["lineage"] = st["lineage"]
        sl["struct"] = st["struct"]
        sl["step"] += 1
        if sl["step"] >= N:
            if learner is not None:
                learner.gae(sl["steps"])
                batch += sl["steps"]
            new_episode(e)
        emit(e)
        if learner is not None and len(batch) >= upd_every:
            s = learner.update(batch)
            s.update(update=learner.n_upd, results=n_res, new_v=n_new, buffer=len(learner.pos),
                     r_mean=round(float(np.mean([x["r"] for x in batch])), 4),
                     modes={m: sum(x["mode"] == i for x in batch) for i, m in enumerate(MODES)},
                     hours=round((time.time() - t0) / 3600, 3), time=time.strftime("%F %T"))
            alog.write(json.dumps(s) + "\n")
            alog.flush()
            batch = []
            if learner.n_upd % 20 == 0:
                torch.save(learner.pol.state_dict(), os.path.join(out, "policy.pt"))
    if learner is not None:
        torch.save(learner.pol.state_dict(), os.path.join(out, "policy.pt"))
    log.close()
    alog.close()


class CtxActor:
    """tcenv.py side (--policy ctx): runs whichever slot's program is ready (FIFO), returns the
    stage and the hardware observation to the agent."""
    needs_all_generators = False
    wants_info = True

    def __init__(self, out, action, opts):
        import multiprocessing as mp
        self.action, self.out, self.opts = dict(action), out, opts
        self.mp = mp.get_context("spawn")
        self.prog_q, self.res_q = self.mp.Queue(), self.mp.Queue()
        self.ready = deque()
        self.proc, self.installed, self.cur, self.table, self.specs = None, {}, None, None, None

    def choose(self):
        return dict(self.action), "ctx"

    def _start(self, gen):
        from rvzr.config import CONF
        fg = gen._function_generator
        isa = fg._isa_spec
        td = fg._instruction_generator._operand_generator.target_desc
        assert CONF.min_bb_per_function == CONF.max_bb_per_function, "random block count unsupported"
        self.table = build_table(isa, td.mem_index_registers, CONF.program_size,
                                 CONF.avg_mem_accesses, CONF.min_bb_per_function)
        self.specs = list(isa.non_memory_access_specs) + list(isa.store_instructions) + \
            list(isa.load_instruction)
        json.dump({k: v for k, v in self.table.items() if k != "slot_prior"},
                  open(os.path.join(self.out, "table.json"), "w"))
        self.proc = self.mp.Process(target=agent_main, daemon=True,
                                    args=(self.table, self.out, self.prog_q, self.res_q, self.opts))
        self.proc.start()

    def before_generate(self, gen):
        if self.proc is None:
            self._start(gen)
        if id(gen) not in self.installed:
            install(gen, self)
            self.installed[id(gen)] = True
        while True:                                   # drain without blocking
            try:
                self.ready.append(self.prog_q.get_nowait())
            except queue.Empty:
                break
        if not self.ready:
            self.ready.append(self.prog_q.get(timeout=600))
        pid, slot, prog, struct = self.ready.popleft()
        self.cur = (pid, prog)
        self.slot = slot
        # Revizor draws the CFG (block successors) and the jumps from `random` before it fills
        # the blocks; seeding it here makes them a function of the program's structure tag, so a
        # RERUN / EDIT gets the reference's CFG and jumps back exactly
        random.seed(f"{struct}:cfg")

    def observe(self, stage, seconds, info=None):
        self.res_q.put((self.cur[0], stage, seconds, info))

    def log_extra(self):
        return {"pid": self.cur[0], "slot": self.slot} if self.cur else {}

    def close(self):
        if self.proc is not None:
            self.res_q.put(None)
            self.proc.join(timeout=120)
            if self.proc.is_alive():
                self.proc.terminate()


class CtxABActor:
    """Same-time A/B: several CtxActor arms (each its own agent process, out subdir, cores) take
    turns test case by test case, so every arm sees the same machine state and time of day."""
    needs_all_generators = False
    wants_info = True

    def __init__(self, out, action, arms):
        self.action, self.arms, self.k = dict(action), [], -1
        for name, opts in arms:
            d = os.path.join(out, name)
            os.makedirs(d, exist_ok=True)
            self.arms.append((name, CtxActor(d, action, opts)))

    def choose(self):
        return dict(self.action), "ctx-ab"

    def before_generate(self, gen):
        self.k = (self.k + 1) % len(self.arms)
        a = self.arms[self.k][1]
        a.installed.pop(id(gen), None)       # point the generator's instruction filling at this arm
        a.before_generate(gen)

    def observe(self, stage, seconds, info=None):
        self.arms[self.k][1].observe(stage, seconds, info)

    def log_extra(self):
        if self.k < 0:
            return {}
        return {**self.arms[self.k][1].log_extra(), "arm": self.arms[self.k][0]}

    def close(self):
        for _, a in self.arms:
            a.close()


# ============================================================================ synthetic test
def cmd_synthetic(args):
    """Planted world, no hardware. A program's partial patterns q = #(store -> same-address-
    register load within 3); a FULL pattern additionally has one of a few rare instructions in
    between and violates with probability 0.5 per run (inputs). Observation: near miss (stage
    fast_path) when q >= 2 or full, obs-filtered when q == 1, else spec-filtered; ex_max = 30 q
    + noise. So the observation says which programs are worth re-running (full but unlucky) or
    editing (many partial patterns), as in the edit pilot. Compares new violations per program
    of the context policy with --fresh-only (mode forced to FRESH = progrl)."""
    import torch
    torch.set_num_threads(4)
    from progrl import _offline_generator
    gen, table, specs, CONF = _offline_generator()
    nm, ns = table["n_nonmem"], table["n_store"]
    rng = np.random.default_rng(args.seed)
    rare = set(rng.choice(nm, size=args.rare, replace=False).tolist())

    def mem_fam(x):
        for sl, f in zip(table["slots"][x[0]], x[1]):
            if sl["kind"] == K_MEM:
                return f
        return None

    def patterns(prog):
        q = full = 0
        for i, a in enumerate(prog):
            if not (nm <= a[0] < nm + ns):
                continue
            fa = mem_fam(a)
            for j in range(i + 1, min(i + 4, len(prog))):
                if prog[j][0] >= nm + ns and fa is not None and mem_fam(prog[j]) == fa:
                    q += 1
                    full += any(prog[k][0] in rare for k in range(i + 1, j))
        return q, full

    pp = PriorPrograms(table, np.random.default_rng(99))
    qs = np.array([patterns(pp.fresh())[0] for _ in range(2000)])
    q_obs, q_near = np.percentile(qs, 40), np.percentile(qs, 90)
    print(f"prior programs: partial patterns median {np.median(qs):.0f}; obs-filter stage from q >= "
          f"{q_obs:.0f}, near miss from q >= {q_near:.0f}", flush=True)

    def run(prog):
        q, full = patterns(prog)
        v = full > 0 and rng.random() < 0.5
        stage = "violation" if v else ("fast_path" if q >= q_near or full else
                                       "observ_filter" if q >= q_obs else "spec_filter")
        info = {"ex_max": int(15 * q + 60 * full + rng.normal(0, 15)), "ex_mean": 5.0 * q, "n_in": 10,
                "ex_pos": min(10, q)}
        if stage != "spec_filter":
            info["fdiv"] = int(stage not in ("observ_filter",))
        return stage, info

    L = CtxLearner(table, seed=args.seed, lr=args.lr, kl=args.kl, sil=args.sil, arch=args.arch,
                   tf={"tf_layers": args.tf_layers} if args.arch == "tf" else None)
    if args.fresh_only:
        with torch.no_grad():
            L.pol.mode_prior.copy_(torch.tensor([0.0, NEG, NEG]))
    E, N = 16, 8
    rewarded, slots, batch = set(), [], []
    hist, modes = deque(maxlen=4000), deque(maxlen=4000)
    rew = deque(maxlen=20000)
    for e in range(E):
        slots.append({"step": 0, "seed": e, "ref": None, "prev": None, "steps": []})
    n, t0 = 0, time.time()
    print(f"{'fresh-only' if args.fresh_only else 'context'} policy; rare set {sorted(rare)}", flush=True)
    while n < args.programs:
        ctxs = np.stack([ctx_vector(s["step"], N, s["prev"]) for s in slots])
        outs = L.sampler.sample(ctxs, [s["ref"] for s in slots])
        for e, (mode, prog, toks, copied) in enumerate(outs):
            sl = slots[e]
            for x in prog:
                if x[3] is None:
                    x[3] = f"{sl['seed']}:{n}"
            stage, info = run(prog)
            h = prog_hash(prog)
            new_v = stage == "violation" and h not in rewarded
            if new_v:
                rewarded.add(h)
            st = {"toks": toks, "ctx": ctxs[e], "r": float(new_v), "mode": mode}
            if new_v:
                L.add_positive(st)
            hist.append(new_v)
            rew.append((stage_class(sl["prev"]["stage"], sl["prev"]["new_v"]) if sl["prev"] else -1, mode, new_v))
            modes.append((mode, stage_class(sl["prev"]["stage"], sl["prev"]["new_v"]) if sl["prev"] else -1))
            sl["steps"].append(st)
            sl["prev"] = {"stage": stage, "info": info, "new_v": new_v, "mode": mode, "copied": copied}
            sl["ref"], sl["step"] = prog, sl["step"] + 1
            n += 1
            if sl["step"] >= N:
                L.gae(sl["steps"])
                batch += sl["steps"]
                slots[e] = {"step": 0, "seed": e * 100003 + n, "ref": None, "prev": None, "steps": []}
        if len(batch) >= 256:
            s = L.update(batch)
            batch = []
            if L.n_upd % args.report == 0:
                by = {}
                for m, c in modes:
                    by.setdefault(c, [0, 0, 0])[m] += 1
                names = {-1: "first", 0: "spec", 1: "obs", 2: "near", 3: "newV", 4: "dupV"}
                mix = "  ".join(f"{names[c]}:" + "/".join(f"{x / max(sum(v), 1):.2f}" for x in v)
                                for c, v in sorted(by.items()))
                print(f"programs {n:7d} ({(time.time() - t0) / 60:5.1f} min) new-V/program "
                      f"{np.mean(hist):.4f}  V {s['v_mean']:.3f} ret {s['ret_mean']:.3f} kl {s['kl']:.3f}  "
                      f"mode fresh/rerun/edit after {mix}", flush=True)
                cell = {}
                for c, m, v in rew:
                    cell.setdefault((c, m), []).append(v)
                print("      new-V rate by (previous outcome, mode): " + "  ".join(
                    f"{names[c]}/{MODES[m]} {np.mean(v):.3f}(n{len(v)})" for (c, m), v in sorted(cell.items())
                    if len(v) >= 20), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--programs", type=int, default=60000)
    s.add_argument("--fresh-only", action="store_true")
    s.add_argument("--rare", type=int, default=3)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--kl", type=float, default=0.1)
    s.add_argument("--sil", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--report", type=int, default=20)
    s.add_argument("--arch", choices=["gru", "tf"], default="gru")
    s.add_argument("--tf-layers", type=int, default=2)
    args = ap.parse_args()
    cmd_synthetic(args)


if __name__ == "__main__":
    main()
