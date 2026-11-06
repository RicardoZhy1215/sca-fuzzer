"""
Summary of a tcenv.py --policy progrl run: violations per hour / per 1000 test cases / test cases
per minute in equal blocks of test-case time, and the agent's training trace (KL to the Revizor
prior, entropy, self-imitation buffer).

  python rvzr/SpecRL/genbandit/analyze_progrl.py rvzr/SpecRL/genbandit/runs/<run> [--blocks 6]
"""
import argparse
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--blocks", type=int, default=6)
args = ap.parse_args()
R = []
for l in open(os.path.join(args.run, "tc.jsonl")):
    try:
        R.append(json.loads(l))
    except json.JSONDecodeError:
        pass
S = os.path.join(args.run, "summary.json")
S = json.load(open(S)) if os.path.exists(S) else {}
tot = sum(r["s"] for r in R)
v = sum(r["stage"] == "violation" for r in R)
print(f"{len(R)} tc, {v} violations, {3600 * v / max(tot, 1e-9):.1f} V/h over test-case time "
      f"({S.get('v_per_h', 'n/a')} V/h wall), {60 * len(R) / max(tot, 1e-9):.0f} tc/min, "
      f"overhead {sum(r['ovh_ms'] for r in R) / max(len(R), 1):.2f} ms/tc")
print("block   hours      tc     V    V/h (95% CI)        V/1000tc  tc/min")
step, acc, i0, b = tot / args.blocks, 0.0, 0, 0
for i, r in enumerate(R):
    acc += r["s"]
    if acc >= step * (b + 1) or i == len(R) - 1:
        blk = R[i0:i + 1]
        s = sum(x["s"] for x in blk)
        vb = sum(x["stage"] == "violation" for x in blk)
        lo, hi = max(vb - 1.96 * math.sqrt(vb), 0), vb + 1.96 * math.sqrt(max(vb, 1))
        print(f"{b + 1:5d} {acc / 3600:7.2f} {len(blk):7d} {vb:5d} {3600 * vb / s:6.1f} "
              f"({3600 * lo / s:4.0f}-{3600 * hi / s:4.0f})   {1000 * vb / len(blk):7.2f}  "
              f"{60 * len(blk) / s:6.0f}")
        i0, b = i + 1, b + 1
A = os.path.join(args.run, "agent.jsonl")
if os.path.exists(A):
    U = [json.loads(l) for l in open(A)]
    print(f"agent: {len(U)} updates; every ~{max(len(U) // 8, 1)}th shown")
    print("update  hours  results  r_mean   kl     ent   buffer  new_pos  upd_s")
    for u in U[::max(len(U) // 8, 1)] + U[-1:]:
        print(f"{u['update'] if 'update' in u else '-':>6} {u.get('hours', 0):6.2f} {u.get('results', 0):8d} "
              f"{u['r_mean']:.4f} {u.get('kl', 0):6.3f} {u.get('ent', 0):6.2f} {u['buffer']:7d} "
              f"{u.get('new_pos', 0):7d} {u.get('upd_s', 0):6.2f}")
