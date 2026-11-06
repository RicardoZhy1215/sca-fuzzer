"""
Summary of a tcenv.py --policy ctx run (rule / fresh / learn): NEW violations (deduplicated by
program) per program and per hour, overall and by mode and by the previous program's outcome;
duplicate violations; speed; the agent's training trace if it learned.

  python rvzr/SpecRL/genbandit/analyze_ctx.py rvzr/SpecRL/genbandit/runs/<run> [--blocks 4]
"""
import argparse
import json
import math
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--blocks", type=int, default=4)
args = ap.parse_args()
R = []
for l in open(os.path.join(args.run, "steps.jsonl")):
    try:
        R.append(json.loads(l))
    except json.JSONDecodeError:
        pass


def rate(rows, key="new_v"):
    n = len(rows)
    if not n:
        return "     -"
    v = sum(bool(r[key]) for r in rows)
    p = v / n
    hw = 1.96 * math.sqrt(max(p * (1 - p), 1 / n) / n)
    return f"{n:6d} {v:5d} {1000 * p:6.1f} ({1000 * max(p - hw, 0):5.1f}-{1000 * (p + hw):5.1f})"


S = sum(r["s"] for r in R)
nv = sum(r["new_v"] for r in R)
dv = sum(r["dup_v"] for r in R)
print(f"{len(R)} programs, {nv} new violations, {dv} duplicate violations, {S / 3600:.2f} h of "
      f"test-case time: {3600 * nv / S:.1f} new V/h, {60 * len(R) / S:.0f} programs/min")
tc = os.path.join(args.run, "tc.jsonl")
if os.path.exists(tc):
    ov = []
    for l in open(tc):
        try:
            ov.append(json.loads(l)["ovh_ms"])
        except Exception:
            pass
    ov.sort()
    print(f"policy overhead per test case: median {ov[len(ov) // 2]:.2f} ms, p99 {ov[int(.99 * len(ov))]:.1f} ms")
print("                               n     V  new V/1000 (95% CI)")
for m in ("fresh", "rerun", "edit"):
    print(f"mode {m:6s}                {rate([r for r in R if r['mode'] == m])}")
# by the previous program's outcome (same episode)
prev = {}
by = defaultdict(list)
for r in sorted(R, key=lambda r: (r["ep"], r["step"])):
    p = prev.get(r["ep"])
    if p is None:
        c = "first"
    elif p["stage"] == "spec_filter":
        c = "spec"
    elif p["stage"] == "observ_filter":
        c = "obs"
    elif p["stage"] == "violation":
        c = "newV" if p["new_v"] else "dupV"
    else:
        c = "near"
    by[(c, r["mode"])].append(r)
    prev[r["ep"]] = r
print("after previous outcome (mode): new V/1000")
for c in ("first", "spec", "obs", "near", "newV", "dupV"):
    for m in ("fresh", "rerun", "edit"):
        if by[(c, m)]:
            print(f"  {c:5s} {m:6s}             {rate(by[(c, m)])}")
# time blocks
step, acc, i0, b = S / args.blocks, 0.0, 0, 0
print("block  hours  programs  new V  new V/h  new V/1000")
for i, r in enumerate(R):
    acc += r["s"]
    if acc >= step * (b + 1) or i == len(R) - 1:
        blk = R[i0:i + 1]
        s = sum(x["s"] for x in blk)
        v = sum(x["new_v"] for x in blk)
        print(f"{b + 1:5d} {acc / 3600:6.2f} {len(blk):9d} {v:6d} {3600 * v / s:8.1f} {1000 * v / len(blk):10.2f}")
        i0, b = i + 1, b + 1
A = os.path.join(args.run, "agent.jsonl")
if os.path.exists(A):
    U = [json.loads(l) for l in open(A)]
    if U:
        print(f"agent: {len(U)} updates")
        for u in U[::max(len(U) // 8, 1)] + U[-1:]:
            print(f"  upd {u['update']:4d} h {u['hours']:5.2f} r_mean {u['r_mean']:.4f} V {u['v_mean']:.3f} "
                  f"ret {u['ret_mean']:.3f} kl {u['kl']:.3f} ent {u['ent']:.2f} modes {u['modes']} "
                  f"buffer {u['buffer']}")
