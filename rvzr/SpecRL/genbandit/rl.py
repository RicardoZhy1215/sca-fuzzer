"""
RL over the generator's configuration for one campaign: --space v1 (Spectre v1, space_v1.py)
or --space v4 (Spectre v4 / store bypass, space_v4.py).

Action = one value per knob of the space (program size, memory-access ratio, GPR pool, input
entropy, #inputs, ...). The episode is a single step: run an unmodified `revizor.py fuzz` with
that configuration for one time slice, reward = violations per executor-hour of the slice.
There is deliberately no state: the old formulation's state (the program prefix) is what made
credit assignment impossible.

Policy: factored softmax, one categorical per knob, REINFORCE with an EMA baseline and a
normalized, clipped advantage; epsilon-uniform exploration per knob. Every --control-every-th
slice runs the space's CONTROL action (SpecRL's v1.yaml / v4.yaml as is, random seeds) as the
Revizor baseline.

  cd ~/sca-fuzzer
  python rvzr/SpecRL/genbandit/rl.py --space v4 --hours 6 --out rvzr/SpecRL/genbandit/runs/v4-<name>
  python rvzr/SpecRL/genbandit/rl.py --space v4 --summary rvzr/SpecRL/genbandit/runs/v4-<name>
(The v1 run runs/v1-rl-261102-2219 was made with this file under its old name rl_v1.py.)
"""
import argparse
import importlib
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, HERE)
from driver import executor_busy, run_config  # noqa: E402

SPEC = os.path.join(ROOT, "base.json")
space = None  # the action-space module, set from --space


class FactoredPolicy:
    def __init__(self, lr: float, eps: float, rng: random.Random):
        self.lr, self.eps, self.rng = lr, eps, rng
        self.logits = {k: [0.0] * len(v) for k, v in space.KNOBS.items()}
        self.baseline, self.scale, self.n = 0.0, 1.0, 0

    def probs(self, k):
        m = max(self.logits[k])
        e = [math.exp(x - m) for x in self.logits[k]]
        z = sum(e)
        return [x / z for x in e]

    def sample(self):
        a = {}
        for k, vals in space.KNOBS.items():
            p = [(1 - self.eps) * q + self.eps / len(vals) for q in self.probs(k)]
            a[k] = vals[self.rng.choices(range(len(vals)), weights=p)[0]]
        return a

    def update(self, a, reward: float):
        if self.n == 0:
            self.baseline = reward
        adv = (reward - self.baseline) / max(self.scale, 1e-6)
        adv = max(-3.0, min(3.0, adv))
        for k, vals in space.KNOBS.items():
            p = self.probs(k)
            i = vals.index(a[k])
            for j in range(len(vals)):
                self.logits[k][j] += self.lr * adv * ((1.0 if j == i else 0.0) - p[j])
        self.scale = 0.8 * self.scale + 0.2 * abs(reward - self.baseline) if self.n else \
            max(abs(reward), 1.0)
        self.baseline = 0.8 * self.baseline + 0.2 * reward
        self.n += 1

    def snapshot(self):
        return {k: [round(x, 3) for x in self.probs(k)] for k in space.KNOBS}


def load(out):
    path = os.path.join(out, "slices.jsonl")
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def main():
    global space
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=["v1", "v4"], default="v1")
    ap.add_argument("--out")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--slice-min", type=float, default=3.0)
    ap.add_argument("--control-every", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--summary", metavar="OUT")
    args = ap.parse_args()
    space = importlib.import_module(f"space_{args.space}")
    if args.summary:
        summary(args.summary)
        return
    assert args.out, "--out is required"
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    pol = FactoredPolicy(args.lr, args.eps, rng)
    recs = load(args.out)
    for r in recs:                                   # resume
        if r["track"] == "policy":
            pol.update(r["action"], r["reward"])
    spent, k = sum(r["hours"] for r in recs), len(recs)
    print(f"[rl-{args.space}] resume at slice {k}, {spent:.2f} h of {args.hours}", flush=True)

    while spent < args.hours:
        if executor_busy():
            print(f"[rl-{args.space}] executor busy -- stopping", flush=True)
            break
        # slice 0 is a control too: calibrates the baseline rate before any learning
        track = "control" if k == 0 or (args.control_every and
                                        k % args.control_every == args.control_every - 1) \
            else "policy"
        a = dict(space.CONTROL) if track == "control" else pol.sample()
        cfg = space.write(a)
        sdir = os.path.join(args.out, f"slice-{k:04d}-{track}-{space.action_name(a)}")
        res = run_config(cfg, os.path.dirname(cfg), SPEC, sdir, int(args.slice_min * 60),
                         a["n_inputs"])
        hours = res["fuzz_seconds"] / 3600
        v = len(res["violations"])
        reward = v / hours if hours else 0.0
        if track == "policy":
            pol.update(a, reward)
        rec = {"slice": k, "track": track, "action": a, "name": space.action_name(a),
               "hours": round(hours, 4), "test_cases": res["test_cases"], "violations": v,
               "reward": round(reward, 2), "stats": res["stats"], "rc": res["rc"],
               "policy": pol.snapshot(), "time": time.strftime("%F %T")}
        with open(os.path.join(args.out, "slices.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[rl-{args.space}] slice {k:3d} {track:7s} tc={res['test_cases']:5d} V={v:3d} "
              f"rate={reward:7.1f}/h  {a}", flush=True)
        spent += hours
        k += 1
    summary(args.out)


def summary(out):
    recs = load(out)
    if not recs:
        print("no slices yet")
        return
    print(f"== {out}: {len(recs)} slices, {sum(r['hours'] for r in recs):.2f} h")
    for track in ("policy", "control"):
        rs = [r for r in recs if r["track"] == track]
        h = sum(r["hours"] for r in rs)
        v = sum(r["violations"] for r in rs)
        tc = sum(r["test_cases"] for r in rs)
        print(f"  {track:7s}: {len(rs):3d} slices {h:5.2f} h  violations {v:5d} ({v / h if h else 0:7.1f}/h)"
              f"  test cases {tc:6d} ({tc / h / 60 if h else 0:5.1f}/min)")
    pol = [r for r in recs if r["track"] == "policy"]
    if pol:
        third = max(1, len(pol) // 3)
        for name, rs in (("first third", pol[:third]), ("last third", pol[-third:])):
            h = sum(r["hours"] for r in rs)
            print(f"  policy {name:11s}: {sum(r['violations'] for r in rs) / h if h else 0:7.1f} violations/h")
        print("  final policy (probability per value):")
        for k, p in pol[-1]["policy"].items():
            print(f"    {k:13s} " + "  ".join(f"{v}:{q:.2f}" for v, q in zip(space.KNOBS[k], p)))


if __name__ == "__main__":
    main()
