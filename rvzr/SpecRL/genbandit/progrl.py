"""
Program-level RL for Revizor: the policy makes the random choices of Revizor's program generator
-- which instruction, which register family for every register operand, which address register
for every memory operand (and which basic block, when there are several) -- one decision at a
time, conditioned on the decisions already made. Nothing runs on the hardware until the whole
program exists; the program then goes through Revizor's normal pipeline once, and the episode's
only reward is 1 if it produced a violation, else 0.

Why this is fast (vs SpecRL, which ran the program on the hardware after every instruction):
  - one hardware pipeline per program, exactly like Revizor;
  - decisions are sampled in a separate agent process (cores 4-7), in batches, ahead of time:
    the fuzzing loop only turns a decision list into instructions (Revizor's own generator code,
    passes, assembler, ELF parser are unchanged).

How it learns from a sparse terminal reward:
  - the policy starts as EXACTLY Revizor's distribution (prior logits + residual heads whose last
    layer is zero), and the generator config starts at the best one found so far (~2% of
    programs violate), so there are positives from the first minute;
  - PPO-clipped policy gradient with a group baseline (advantage = reward - batch mean; no critic);
  - self-imitation: every violating program is kept (deduplicated) and its likelihood is
    increased at each update, so each rare positive is reused many times;
  - KL to the Revizor prior: decisions unrelated to violations are pulled back to Revizor's
    distribution, which keeps generating variants instead of collapsing on one program.

  python rvzr/SpecRL/genbandit/progrl.py equiv       # prior policy == Revizor's generator? speed?
  python rvzr/SpecRL/genbandit/progrl.py synthetic   # learns a planted rare pattern from 0/1 reward?
  (tcenv.py --policy progrl runs it on the hardware)
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, HERE)

FAMS = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi"]
ALIASES = {"rax": ["rax", "eax", "ax", "al", "ah"], "rbx": ["rbx", "ebx", "bx", "bl", "bh"],
           "rcx": ["rcx", "ecx", "cx", "cl", "ch"], "rdx": ["rdx", "edx", "dx", "dl", "dh"],
           "rsi": ["rsi", "esi", "si", "sil"], "rdi": ["rdi", "edi", "di", "dil"]}
FAMILY = {n: FAMS.index(f) for f, ns in ALIASES.items() for n in ns}
K_SPEC, K_REG, K_MEM, K_BB = 0, 1, 2, 3
NEG = -1e9


# ============================================================================ program space
def build_table(isa, mem_index_registers, program_size: int, avg_mem_accesses: int,
                n_bb: int = 1) -> dict:
    """Plain-data description of the decision space, built from the generator's own instruction
    set (env process) and shipped to the agent. Spec order: non-memory, stores, loads -- the
    three lists Revizor's pick_spec draws from; prior = pick_spec's probabilities.

    A register/memory operand becomes a decision ("slot") only if all of its candidate names
    belong to the 6 GPR families and at least 2 families are possible; the decision picks a
    family, the name inside the family is then drawn uniformly, and the family prior is the
    share of names per family -- so the prior policy reproduces Revizor's choice exactly."""
    nm, st, ld = isa.non_memory_access_specs, isa.store_instructions, isa.load_instruction
    specs = list(nm) + list(st) + list(ld)
    p_mem = min(max(avg_mem_accesses / program_size, 0.0), 1.0) if (st or ld) else 0.0
    prior = []
    for s in nm:
        prior.append((1 - p_mem) / len(nm))
    for s in st:
        prior.append(p_mem * (0.5 if ld else 1.0) / len(st))
    for s in ld:
        prior.append(p_mem * (0.5 if st else 1.0) / len(ld))
    slots, slot_prior = [], []
    for s in specs:
        sl = []
        ops = [(j, False, o) for j, o in enumerate(s.operands)] + \
              [(j, True, o) for j, o in enumerate(s.implicit_operands)]
        for j, implicit, o in ops:
            kind = o.type.name
            if kind == "REG":
                names = list(o.values)
            elif kind == "MEM":
                names = list(o.values) if o.values else list(mem_index_registers)
            else:
                continue
            if not names or any(n not in FAMILY for n in names):
                continue
            counts = [0] * len(FAMS)
            for n in names:
                counts[FAMILY[n]] += 1
            if sum(c > 0 for c in counts) < 2:
                continue
            tot = sum(counts)
            sl.append({"op": j, "implicit": implicit, "kind": K_REG if kind == "REG" else K_MEM,
                       "src": int(bool(o.src)), "dest": int(bool(o.dest)), "sid": len(slot_prior),
                       "pos": min(len(sl), 3)})
            slot_prior.append([math.log(c / tot) if c else NEG for c in counts])
        slots.append(sl)
    names = [f"{s.name} " + ",".join(o.type.name for o in s.operands) for s in specs]
    return {"names": names, "spec_logprior": [math.log(p) for p in prior], "slots": slots,
            "slot_prior": slot_prior, "program_size": int(program_size), "n_bb": int(n_bb),
            "n_store": len(st), "n_load": len(ld), "n_nonmem": len(nm),
            "sig": hashlib.sha1("|".join(names).encode()).hexdigest()[:12]}


# ============================================================================ policy
def make_policy(table: dict, hidden: int = 128, emb: int = 64, seed: int = 0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    n_specs, n_bb = len(table["names"]), table["n_bb"]

    class Policy(nn.Module):
        """GRU over the decisions made so far. Input at step t = embedding of the previous
        decision's value + embedding of the current query (decision kind, operand position,
        src/dest flags, current instruction). Logits = Revizor prior + residual head; every
        head's last layer starts at zero, so a fresh policy IS the prior."""

        def __init__(self):
            super().__init__()
            self.off = {K_SPEC: 1, K_REG: 1 + n_specs, K_MEM: 1 + n_specs + 6,
                        K_BB: 1 + n_specs + 12}
            self.val = nn.Embedding(1 + n_specs + 12 + n_bb, emb)        # 0 = BOS
            self.q_kind = nn.Embedding(4, emb)
            self.q_pos = nn.Embedding(4, emb)
            self.q_sd = nn.Embedding(4, emb)
            self.q_spec = nn.Embedding(n_specs + 1, emb)                 # n_specs = none
            self.gru = nn.GRU(emb, hidden, batch_first=True)
            self.heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                                      nn.Linear(hidden, n))
                                        for n in (n_specs, 6, 6, max(n_bb, 1))])
            for h in self.heads:
                nn.init.zeros_(h[-1].weight)
                nn.init.zeros_(h[-1].bias)
            self.register_buffer("spec_prior", torch.tensor(table["spec_logprior"]))
            sp = table["slot_prior"] or [[0.0] * 6]
            self.register_buffer("slot_prior", torch.tensor(sp, dtype=torch.float32))
            self.register_buffer("bb_prior", torch.zeros(max(n_bb, 1)))

        def query(self, kind, pos, sd, spec):
            return self.q_kind(kind) + self.q_pos(pos) + self.q_sd(sd) + self.q_spec(spec)

        def prior(self, kind: int, sid):
            if kind == K_SPEC:
                return self.spec_prior.expand(len(sid), -1)
            if kind in (K_REG, K_MEM):
                return self.slot_prior[sid]
            return self.bb_prior.expand(len(sid), -1)

        def cell(self, x, h):
            g = self.gru
            gi = x @ g.weight_ih_l0.T + g.bias_ih_l0
            gh = h @ g.weight_hh_l0.T + g.bias_hh_l0
            ir, iz, i_n = gi.chunk(3, 1)
            hr, hz, hn = gh.chunk(3, 1)
            r = torch.sigmoid(ir + hr)
            z = torch.sigmoid(iz + hz)
            n = torch.tanh(i_n + r * hn)
            return (1 - z) * n + z * h

    return Policy()


class Sampler:
    """Batched autoregressive sampling. A program = list of instructions
    (spec index, [family per slot], bb); its tokens (kind, value, query, sid) and per-token
    log-probs are kept for training."""

    def __init__(self, table: dict, policy, rng: np.random.Generator):
        self.t, self.pol, self.rng = table, policy, rng
        self.n_specs = len(table["names"])

    def sample(self, B: int):
        import torch
        t, pol = self.t, self.pol
        P, nbb = t["program_size"], t["n_bb"]
        H = pol.gru.hidden_size
        h = torch.zeros(B, H)
        prev = np.zeros(B, np.int64)                    # value-embedding id of previous token
        inst = np.zeros(B, np.int64)
        phase = np.full(B, -1, np.int64)                # -1 spec, j>=0 slot j, -2 bb
        cur = np.full(B, self.n_specs, np.int64)
        progs = [[] for _ in range(B)]
        toks = [[] for _ in range(B)]
        done = np.zeros(B, bool)
        with torch.no_grad():
            while not done.all():
                act = np.nonzero(~done)[0]
                kind = np.empty(len(act), np.int64)
                pos = np.zeros(len(act), np.int64)
                sd = np.zeros(len(act), np.int64)
                qspec = np.full(len(act), self.n_specs, np.int64)
                sid = np.zeros(len(act), np.int64)
                for k, b in enumerate(act):
                    if phase[b] == -1:
                        kind[k] = K_SPEC
                    elif phase[b] == -2:
                        kind[k], qspec[k] = K_BB, cur[b]
                    else:
                        s = t["slots"][cur[b]][phase[b]]
                        kind[k], pos[k], sd[k], qspec[k], sid[k] = \
                            s["kind"], s["pos"], 2 * s["src"] + s["dest"], cur[b], s["sid"]
                T = torch.from_numpy
                x = pol.val(T(prev[act])) + pol.query(T(kind), T(pos), T(sd), T(qspec))
                ha = pol.cell(x, h[act])
                h[act] = ha
                val = np.empty(len(act), np.int64)
                lp = np.empty(len(act), np.float32)
                for kd in np.unique(kind):
                    m = np.nonzero(kind == kd)[0]
                    lg = pol.prior(int(kd), T(sid[m])) + pol.heads[kd](ha[m])
                    lsm = torch.log_softmax(lg, 1).numpy()
                    pr = np.exp(lsm)
                    u = self.rng.random((len(m), 1))
                    v = np.minimum((pr.cumsum(1) < u).sum(1), pr.shape[1] - 1)
                    v = np.where(lsm[np.arange(len(m)), v] < -1e8, pr.argmax(1), v)  # fp guard
                    val[m] = v
                    lp[m] = lsm[np.arange(len(m)), v]
                for k, b in enumerate(act):
                    kd, v = int(kind[k]), int(val[k])
                    toks[b].append((kd, v, int(pos[k]), int(sd[k]), int(qspec[k]), int(sid[k]),
                                    float(lp[k]), int(prev[b])))
                    prev[b] = pol.off[kd] + v
                    if kd == K_SPEC:
                        cur[b] = v
                        progs[b].append([v, [], 0])
                        nxt = 0 if t["slots"][v] else (-2 if nbb > 1 else None)
                    elif kd == K_BB:
                        progs[b][-1][2] = v
                        nxt = None
                    else:
                        progs[b][-1][1].append(v)
                        j = phase[b] + 1
                        nxt = j if j < len(t["slots"][cur[b]]) else (-2 if nbb > 1 else None)
                    if nxt is None:
                        inst[b] += 1
                        if inst[b] >= P:
                            done[b] = True
                        nxt = -1
                    phase[b] = nxt
        return progs, toks


def evaluate(pol, toks_batch):
    """Teacher-forced log-prob and KL-to-prior of every token. toks: list of token lists."""
    import torch
    B = len(toks_batch)
    T = max(len(x) for x in toks_batch)
    arr = np.zeros((B, T, 8), np.float64)
    valid = np.zeros((B, T), bool)
    for i, x in enumerate(toks_batch):
        arr[i, :len(x)] = np.asarray(x, np.float64)
        valid[i, :len(x)] = True
    a = torch.from_numpy(arr)
    kind, val, pos, sd, qs, sid, prev = (a[..., c].long() for c in (0, 1, 2, 3, 4, 5, 7))
    x = pol.val(prev) + pol.query(kind, pos, sd, qs)
    h, _ = pol.gru(x)
    logp = torch.zeros(B, T)
    kl = torch.zeros(B, T)
    ent = torch.zeros(B, T)
    vt = torch.from_numpy(valid)
    for kd in range(4):
        m = (kind == kd) & vt
        if not m.any():
            continue
        hm = h[m]
        pri = pol.prior(kd, sid[m])
        lg = pri + pol.heads[kd](hm)
        lsm = torch.log_softmax(lg, 1)
        lsm0 = torch.log_softmax(pri, 1)
        pr = lsm.exp()
        ok = pri > NEG / 2
        logp[m] = lsm.gather(1, val[m].unsqueeze(1)).squeeze(1)
        kl[m] = torch.where(ok, pr * (lsm - lsm0), torch.zeros_like(pr)).sum(1)
        ent[m] = -torch.where(ok, pr * lsm, torch.zeros_like(pr)).sum(1)
    return logp, kl, ent, vt, torch.from_numpy(arr[..., 6]).float()


# ============================================================================ learner
class Learner:
    def __init__(self, table, seed=0, lr=3e-4, kl=0.1, sil=1.0, clip=0.2, epochs=2,
                 sil_batch=64, buf_cap=5000, frozen=False):
        import torch
        self.torch = torch
        self.pol = make_policy(table, seed=seed)
        self.opt = torch.optim.Adam(self.pol.parameters(), lr=lr)
        self.sampler = Sampler(table, self.pol, np.random.default_rng(seed + 1))
        self.kl, self.sil, self.clip, self.epochs = kl, sil, clip, epochs
        self.sil_batch, self.buf_cap, self.frozen = sil_batch, buf_cap, frozen
        self.pos, self.pos_keys = [], set()
        self.rng = random.Random(seed)
        self.n_upd = 0

    @staticmethod
    def key(prog):
        return hashlib.sha1(json.dumps(prog).encode()).hexdigest()

    def add_positive(self, prog, toks):
        k = self.key(prog)
        if k in self.pos_keys:
            return False
        self.pos_keys.add(k)
        self.pos.append(toks)
        if len(self.pos) > self.buf_cap:
            self.pos.pop(0)
        return True

    def update(self, batch):
        """batch: list of (toks, reward). Returns stats."""
        torch = self.torch
        t0 = time.time()
        r = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        adv = r - r.mean()
        toks = [b[0] for b in batch]
        st = {"n": len(batch), "r_mean": float(r.mean()), "buffer": len(self.pos)}
        if self.frozen:
            with torch.no_grad():
                _, kl, ent, vt, _ = evaluate(self.pol, toks)
            st.update(kl=float(kl[vt].mean()), ent=float(ent[vt].mean()), upd_s=time.time() - t0)
            return st
        for _ in range(self.epochs):
            logp, kl, ent, vt, old = evaluate(self.pol, toks)
            ratio = torch.exp(logp - old)
            A = adv[:, None].expand_as(ratio)
            pg = -torch.minimum(ratio * A, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * A)
            pg = pg[vt].mean()
            klm = kl[vt].mean()
            loss = pg + self.kl * klm
            sil = torch.tensor(0.0)
            if self.pos and self.sil:
                sb = self.rng.sample(self.pos, min(self.sil_batch, len(self.pos)))
                lps, _, _, vts, _ = evaluate(self.pol, sb)
                sil = -lps[vts].mean()
                loss = loss + self.sil * sil
            self.opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.pol.parameters(), 1.0)
            self.opt.step()
        self.n_upd += 1
        st.update(pg=float(pg), kl=float(klm), sil=float(sil), ent=float(ent[vt].mean()),
                  grad=float(gn), upd_s=round(time.time() - t0, 2))
        return st


# ============================================================================ agent process
def agent_main(table, out, prog_q, res_q, opts):
    """Runs on cores 4-7: keeps prog_q filled with sampled programs, trains on the rewards."""
    try:
        os.sched_setaffinity(0, set(opts.get("cores", [4, 5, 6, 7])))
    except Exception:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(4)
    L = Learner(table, seed=opts.get("seed", 0), lr=opts.get("lr", 3e-4), kl=opts.get("kl", 0.1),
                sil=opts.get("sil", 1.0), frozen=opts.get("frozen", False))
    log = open(os.path.join(out, "agent.jsonl"), "a")
    pending, batch, pid = {}, [], 0
    nb, upd_every, depth = opts.get("sample_batch", 64), opts.get("update_every", 256), \
        opts.get("queue_batches", 4)
    n_res, n_pos_new, t_start = 0, 0, time.time()
    while True:
        while prog_q.qsize() < depth:
            progs, toks = L.sampler.sample(nb)
            items = []
            for p, tk in zip(progs, toks):
                pending[pid] = (p, tk)
                items.append((pid, p))
                pid += 1
            prog_q.put(items)
        try:
            msg = res_q.get(timeout=0.05)
        except queue.Empty:
            continue
        if msg is None:
            break
        rid, reward, stage, secs = msg
        p, tk = pending.pop(rid)
        n_res += 1
        if reward > 0 and L.add_positive(p, tk):
            n_pos_new += 1
        batch.append((tk, reward))
        if len(batch) >= upd_every:
            st = L.update(batch)
            st.update(update=L.n_upd, results=n_res, new_pos=n_pos_new,
                      hours=round((time.time() - t_start) / 3600, 3), time=time.strftime("%F %T"))
            log.write(json.dumps(st) + "\n")
            log.flush()
            batch = []
            if L.n_upd % 20 == 0:
                torch.save(L.pol.state_dict(), os.path.join(out, "policy.pt"))
    torch.save(L.pol.state_dict(), os.path.join(out, "policy.pt"))
    log.close()


class ProgActor:
    """tcenv.py side (--policy progrl): fixed generator config; the program itself comes from
    the agent. Hooks: choose() -> config, before_generate(gen) installs the decision list,
    observe(stage, seconds) returns the reward, close() stops the agent."""
    needs_all_generators = False

    def __init__(self, out, action, opts):
        import multiprocessing as mp
        self.action, self.out, self.opts = dict(action), out, opts
        self.ctx = mp.get_context("spawn")
        self.prog_q, self.res_q = self.ctx.Queue(), self.ctx.Queue()
        self.local, self.cur, self.proc, self.installed = [], None, None, {}
        self.specs = None

    def choose(self):
        return dict(self.action), "rl"

    def _start(self, gen):
        from rvzr.config import CONF
        fg = gen._function_generator
        isa = fg._isa_spec
        n_bb = CONF.min_bb_per_function
        assert CONF.min_bb_per_function == CONF.max_bb_per_function, "random BB count unsupported"
        td = fg._instruction_generator._operand_generator.target_desc
        self.table = build_table(isa, td.mem_index_registers, CONF.program_size,
                                 CONF.avg_mem_accesses, n_bb)
        self.specs = list(isa.non_memory_access_specs) + list(isa.store_instructions) + \
            list(isa.load_instruction)
        json.dump({k: v for k, v in self.table.items() if k != "slot_prior"},
                  open(os.path.join(self.out, "table.json"), "w"))
        self.proc = self.ctx.Process(target=agent_main, daemon=True,
                                     args=(self.table, self.out, self.prog_q, self.res_q, self.opts))
        self.proc.start()

    def before_generate(self, gen):
        if self.proc is None:
            self._start(gen)
        if id(gen) not in self.installed:
            install(gen, self)
            self.installed[id(gen)] = True
        if not self.local:
            self.local = list(self.prog_q.get(timeout=600))
        self.cur = self.local.pop(0)

    def observe(self, stage, seconds):
        self.res_q.put((self.cur[0], 1.0 if stage == "violation" else 0.0, stage, seconds))

    def log_extra(self):
        return {"pid": self.cur[0]} if self.cur else {}

    def close(self):
        if self.proc is not None:
            self.res_q.put(None)
            self.proc.join(timeout=120)
            if self.proc.is_alive():
                self.proc.terminate()


class EditProbeActor:
    """Pilot for "write a program, then revise it from the hardware result" (tcenv.py
    --policy edit). No learning: it measures whether hardware feedback is worth acting on.

    Episode = a fresh program drawn from Revizor's distribution (the prior), run once; its
    outcome puts it in a category (spec-filtered / observation-filtered / near miss = passed
    the observation filter without a violation / violation). Then K[category] follow-up test
    cases, each one of
      rerun    - the same program again (only the inputs are new): the control
      edit_ins - the base program with ONE instruction redrawn from the prior (a different one)
      edit_op  - the base program with ONE register / address-register choice changed
    (star design: every follow-up edits the base, so each is a single, attributable change).
    Every test case logs the observation that would drive an RL editor: stage, excess ops
    (dispatched - retired, per input), number of inputs whose fenced/unfenced traces differ."""
    needs_all_generators = False
    wants_info = True
    K = {"spec": 1, "obs": 2, "near": 6, "viol": 4}

    def __init__(self, out, action, opts):
        self.action, self.out = dict(action), out
        self.rng = np.random.default_rng(opts.get("seed", 1))
        self.p_rerun = opts.get("p_rerun", 0.25)
        self.table, self.specs, self.installed = None, None, {}
        self.cur, self.seed = None, None
        self.ep, self.base, self.base_seed, self.cat, self.left, self.step = -1, None, None, None, 0, 0
        self.kind, self.edit = None, None
        self.log = open(os.path.join(out, "edits.jsonl"), "a", buffering=1 << 14)

    def choose(self):
        return dict(self.action), "edit"

    def _setup(self, gen):
        from rvzr.config import CONF
        fg = gen._function_generator
        isa = fg._isa_spec
        td = fg._instruction_generator._operand_generator.target_desc
        self.table = build_table(isa, td.mem_index_registers, CONF.program_size,
                                 CONF.avg_mem_accesses, CONF.min_bb_per_function)
        assert self.table["n_bb"] == 1, "edit pilot: single basic block only"
        self.specs = list(isa.non_memory_access_specs) + list(isa.store_instructions) + \
            list(isa.load_instruction)
        self.p_spec = np.exp(np.array(self.table["spec_logprior"]))
        self.p_spec /= self.p_spec.sum()
        self.p_slot = [np.exp(np.array(sp)) / np.exp(np.array(sp)).sum()
                       for sp in self.table["slot_prior"]]
        json.dump({k: v for k, v in self.table.items() if k != "slot_prior"},
                  open(os.path.join(self.out, "table.json"), "w"))

    def _inst(self, spec=None):
        s = int(self.rng.choice(len(self.p_spec), p=self.p_spec)) if spec is None else spec
        fams = [int(self.rng.choice(6, p=self.p_slot[sl["sid"]])) for sl in self.table["slots"][s]]
        return [s, fams, 0]

    def _edited(self):
        prog = [[s, list(f), b] for s, f, b in self.base]
        if self.rng.random() < 0.5:
            pos = int(self.rng.integers(len(prog)))
            old = prog[pos][0]
            new = old
            while new == old:
                new = int(self.rng.choice(len(self.p_spec), p=self.p_spec))
            prog[pos] = self._inst(new)
            return prog, "edit_ins", {"pos": pos, "from": old, "to": new}
        slots = [(i, j) for i, (s, f, _) in enumerate(prog) for j in range(len(f))]
        i, j = slots[int(self.rng.integers(len(slots)))]
        sl = self.table["slots"][prog[i][0]][j]
        p = self.p_slot[sl["sid"]].copy()
        old = prog[i][1][j]
        p[old] = 0
        if p.sum() == 0:
            return prog, "rerun", {}
        new = int(self.rng.choice(6, p=p / p.sum()))
        prog[i][1][j] = new
        return prog, "edit_op", {"pos": i, "slot": j, "from": old, "to": new,
                                 "slot_kind": "mem" if sl["kind"] == K_MEM else "reg"}

    def before_generate(self, gen):
        if self.table is None:
            self._setup(gen)
        if id(gen) not in self.installed:
            install(gen, self)
            self.installed[id(gen)] = True
        if self.left <= 0:
            self.ep += 1
            self.step = 0
            self.base = [self._inst() for _ in range(self.table["program_size"])]
            self.base_seed = int(self.rng.integers(1 << 62))
            self.cat, self.kind, self.edit, prog = None, "fresh", {}, self.base
        else:
            self.step += 1
            self.left -= 1
            if self.rng.random() < self.p_rerun:
                prog, self.kind, self.edit = self.base, "rerun", {}
            else:
                prog, self.kind, self.edit = self._edited()
        self.seed = self.base_seed
        self.cur = (self.ep, prog)

    def observe(self, stage, seconds, info=None):
        if self.kind == "fresh":
            self.cat = {"violation": "viol", "spec_filter": "spec",
                        "observ_filter": "obs"}.get(stage, "near")
            self.left = self.K[self.cat]
        rec = {"ep": self.ep, "step": self.step, "kind": self.kind, "cat": self.cat,
               "stage": stage, "s": round(seconds, 4), **(info or {}), **self.edit}
        self.log.write(json.dumps(rec) + "\n")

    def log_extra(self):
        return {"ep": self.ep, "step": self.step, "kind": self.kind}

    def close(self):
        self.log.close()


def install(gen, source):
    """Replace the generator's instruction filling with one that follows source.cur's
    decisions; everything else (CFG, terminators, passes, printing, assembly) is Revizor's."""
    import types
    from rvzr.tc_components.instruction import Instruction, RegisterOp, MemoryOp
    fg = gen._function_generator
    opgen = fg._instruction_generator._operand_generator
    td = opgen.target_desc
    table_slots = None

    def fill(self, func):
        nonlocal table_slots
        if table_slots is None:
            table_slots = source.table["slots"]
        bbs = list(func)
        seed = getattr(source, "seed", None)
        for pos, inst_d in enumerate(source.cur[1]):
            spec_i, fams, bb_i = inst_d[0], inst_d[1], inst_d[2]
            tag = inst_d[3] if len(inst_d) > 3 else None
            if tag is not None:
                # ctxrl: the instruction's origin (episode seed + the step that created it) --
                # a copied instruction keeps its tag, so it is reproduced exactly
                random.seed(f"{tag}:{pos}:{spec_i}")
            elif seed is not None:
                # the parts Revizor still draws at random (immediates, the name inside a register
                # family, lea's address operands) depend only on (program, position, instruction):
                # re-running a program reproduces it exactly, editing one instruction leaves the
                # others' random parts unchanged
                random.seed(f"{seed}:{pos}:{spec_i}")
            spec = source.specs[spec_i]
            inst = Instruction.from_spec(spec)
            by_op = {(s["op"], s["implicit"]): f for s, f in zip(table_slots[spec_i], fams)}
            for implicit, ops in ((False, spec.operands), (True, spec.implicit_operands)):
                for j, o in enumerate(ops):
                    f = by_op.get((j, implicit))
                    if f is None:
                        op = opgen.generate(o, inst)
                    elif o.type.name == "REG":
                        op = RegisterOp(random.choice([n for n in o.values if FAMILY[n] == f]),
                                        o.width, o.src, o.dest)
                    else:
                        names = list(o.values) if o.values else list(td.mem_index_registers)
                        op = MemoryOp(random.choice([n for n in names if FAMILY[n] == f]),
                                      o.width, o.src, o.dest)
                    (inst.implicit_operands if implicit else inst.operands).append(op)
            bb = bbs[bb_i]
            bb.insert_after(bb.get_last(), inst)
        if seed is not None:
            random.seed(f"{seed}:passes")       # the passes' few random draws, fixed per program

    fg._add_instructions_in_function = types.MethodType(fill, fg)


# ============================================================================ offline tests
def _offline_generator(action=None):
    """A generator for the action's config without an executor (as space_v4.validate does)."""
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import space_v4 as space
    from rvzr.config import CONF
    from rvzr.arch.x86.fuzzer import _update_instruction_list
    from rvzr.isa_spec import InstructionSet
    from rvzr.factory import get_program_generator
    if action is None:
        action = json.load(open(os.path.join(HERE, "runs", "coupled-v4-261104-0622",
                                             "policy.json")))["incumbent"]
    cfg = space.write(action)
    CONF.load(cfg, os.path.dirname(cfg))
    _update_instruction_list()
    isa = InstructionSet(space.SPEC, CONF.instruction_categories)
    gen = get_program_generator(CONF.program_generator_seed, isa)
    fg = gen._function_generator
    td = fg._instruction_generator._operand_generator.target_desc
    table = build_table(isa, td.mem_index_registers, CONF.program_size, CONF.avg_mem_accesses,
                        CONF.min_bb_per_function)
    specs = list(isa.non_memory_access_specs) + list(isa.store_instructions) + \
        list(isa.load_instruction)
    return gen, table, specs, CONF


def _program_stats(test_case, names_index):
    """(spec-ish key per instruction, address family per memory operand, family per GPR operand)"""
    sk, mem, reg = [], [], []
    for bb in test_case.iter_basic_blocks():
        for inst in bb:
            if inst.is_instrumentation or inst.category == "MACRO" or inst.is_control_flow:
                continue
            sk.append(inst.name + " " + ",".join(type(o).__name__[:3] for o in inst.operands))
            for o in inst.operands:
                if type(o).__name__ == "MemoryOp":
                    for tok in o.value.replace("+", " ").split():
                        if tok in FAMILY:
                            mem.append(FAMILY[tok])
                elif type(o).__name__ == "RegisterOp" and o.value in FAMILY:
                    reg.append(FAMILY[o.value])
    return sk, mem, reg


def cmd_equiv(args):
    """Prior policy vs Revizor's own generator: same distribution? generation time?"""
    import torch
    from collections import Counter
    torch.set_num_threads(1)
    gen, table, specs, CONF = _offline_generator()
    n_slots = sum(len(s) for s in table["slots"])
    print(f"specs {len(specs)} (non-mem {table['n_nonmem']}, store {table['n_store']}, "
          f"load {table['n_load']}); decision slots {n_slots}; program_size "
          f"{table['program_size']}; p_mem {CONF.avg_mem_accesses / CONF.program_size:.3f}")
    asm = "/tmp/claude-1000/progrl_equiv.asm"
    os.makedirs(os.path.dirname(asm), exist_ok=True)
    N = args.n
    stats = {}
    t = time.perf_counter()
    for _ in range(N):
        tc = gen.create_test_case(asm)
        s = _program_stats(tc, None)
        for k, v in zip(("spec", "mem", "reg"), s):
            stats.setdefault(("revizor", k), Counter()).update(v)
    t_rev = (time.perf_counter() - t) / N
    L = Learner(table, frozen=True)

    class Src:
        pass
    src = Src()
    src.table, src.specs = table, specs
    install(gen, src)
    t_samp = 0.0
    t = time.perf_counter()
    lp_err = 0.0
    for i in range(0, N, 64):
        t1 = time.perf_counter()
        progs, toks = L.sampler.sample(min(64, N - i))
        t_samp += time.perf_counter() - t1
        # prior check: sampled log-probs == teacher-forced == log prior of the choices
        if i == 0:
            with torch.no_grad():
                lp, _, _, vt, old = evaluate(L.pol, toks)
            lp_err = float((lp[vt] - old[vt]).abs().max())
        for p in progs:
            src.cur = (0, p)
            tc = gen.create_test_case(asm)
            s = _program_stats(tc, None)
            for k, v in zip(("spec", "mem", "reg"), s):
                stats.setdefault(("policy", k), Counter()).update(v)
    t_pol = (time.perf_counter() - t - t_samp) / N
    print(f"create_test_case: revizor {1000 * t_rev:.2f} ms, policy-driven {1000 * t_pol:.2f} ms; "
          f"sampling {1000 * t_samp / N:.3f} ms/program (batch 64, 1 thread)")
    print(f"max |sampled logp - teacher-forced logp| = {lp_err:.2e}")
    from scipy.stats import chi2_contingency
    for k in ("spec", "mem", "reg"):
        a, b = stats[("revizor", k)], stats[("policy", k)]
        keys = sorted(set(a) | set(b))
        tab = np.array([[a[x] for x in keys], [b[x] for x in keys]])
        keep = tab.sum(0) >= 10   # chi2 needs non-tiny cells; pool the rest
        tab2 = np.concatenate([tab[:, keep], tab[:, ~keep].sum(1, keepdims=True)], 1) \
            if (~keep).any() else tab[:, keep]
        tab2 = tab2[:, tab2.sum(0) > 0]
        chi, p, dof, _ = chi2_contingency(tab2)
        tv = 0.5 * np.abs(tab[0] / tab[0].sum() - tab[1] / tab[1].sum()).sum()
        print(f"  {k:5s}: {len(keys)} categories, n = {tab[0].sum()} vs {tab[1].sum()}, "
              f"total variation {tv:.3f}, chi2 p = {p:.3f}")


def cmd_synthetic(args):
    """Planted rule on the decisions alone (no assembly, no hardware): reward 1 if some store
    through address family F is followed within 3 instructions by a load through the same
    family F AND a specific rare spec appears in between -- tuned to a ~1% base rate, then each
    positive is rewarded only with probability --hit (input randomness). Does the learner raise
    the rate, and does it keep the programs diverse?"""
    import torch
    torch.set_num_threads(4)
    gen, table, specs, CONF = _offline_generator()
    nm, ns = table["n_nonmem"], table["n_store"]
    is_store = lambda s: nm <= s < nm + ns
    is_load = lambda s: s >= nm + ns
    rng = np.random.default_rng(args.seed)
    rare = set(rng.choice(nm, size=args.rare, replace=False).tolist())

    def mem_fam(prog_inst):
        s, fams, _ = prog_inst
        for sl, f in zip(table["slots"][s], fams):
            if sl["kind"] == K_MEM:
                return f
        return None

    def reward(prog):
        for i, a in enumerate(prog):
            if not is_store(a[0]):
                continue
            fa = mem_fam(a)
            for j in range(i + 1, min(i + 4, len(prog))):
                if is_load(prog[j][0]) and mem_fam(prog[j]) == fa and fa is not None and \
                        any(prog[k][0] in rare for k in range(i + 1, j)):
                    return 1.0
        return 0.0

    L = Learner(table, seed=args.seed, lr=args.lr, kl=args.kl, sil=args.sil)
    base = np.mean([reward(p) for p in L.sampler.sample(512)[0]])
    print(f"planted rule base rate (prior policy): {base:.4f}; rewarded with prob {args.hit}")
    batch, n, hits, t0 = [], 0, [], time.time()
    distinct = set()
    while n < args.programs:
        progs, toks = L.sampler.sample(64)
        for p, tk in zip(progs, toks):
            true = reward(p)
            r = true if rng.random() < args.hit else 0.0
            hits.append(true)
            n += 1
            if r > 0:
                L.add_positive(p, tk)
                distinct.add(Learner.key(p))
            batch.append((tk, r))
            if len(batch) >= 256:
                st = L.update(batch)
                batch = []
                if L.n_upd % args.report == 0:
                    rate = np.mean(hits[-2560:])
                    print(f"programs {n:7d} ({(time.time() - t0) / 60:5.1f} min): true rate "
                          f"{rate:.4f} (x{rate / max(base, 1e-9):.1f}); kl {st['kl']:.3f} ent "
                          f"{st['ent']:.2f} buffer {st['buffer']} distinct positives "
                          f"{len(distinct)}", flush=True)
    torch.save(L.pol.state_dict(), os.path.join(args.out, "synthetic_policy.pt")) if args.out else None


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("equiv")
    e.add_argument("--n", type=int, default=1024)
    s = sub.add_parser("synthetic")
    s.add_argument("--programs", type=int, default=100000)
    s.add_argument("--rare", type=int, default=3, help="size of the rare non-memory spec set")
    s.add_argument("--hit", type=float, default=0.5)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--kl", type=float, default=0.1)
    s.add_argument("--sil", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--report", type=int, default=10)
    s.add_argument("--out", default="")
    args = ap.parse_args()
    {"equiv": cmd_equiv, "synthetic": cmd_synthetic}[args.cmd](args)


if __name__ == "__main__":
    main()
