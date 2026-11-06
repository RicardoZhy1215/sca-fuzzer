"""
Summary of a tcenv.py --policy coupled run: per track (incumbent / candidate / explore) test
cases, violations, V/h; overall V/h vs the incumbent's own interleaved V/h (the cost or gain of
exploring); incumbent changes; policy overhead.

  python rvzr/SpecRL/genbandit/analyze_coupled.py rvzr/SpecRL/genbandit/runs/<run>
"""
import json
import math
import os
import sys
from collections import defaultdict

run = sys.argv[1]
R = []
for l in open(os.path.join(run, "tc.jsonl")):
    try:
        R.append(json.loads(l))
    except json.JSONDecodeError:
        pass
S = json.load(open(os.path.join(run, "summary.json"))) if os.path.exists(os.path.join(run, "summary.json")) else {}
tr = defaultdict(lambda: [0, 0, 0.0])
cfg = defaultdict(lambda: [0, 0, 0.0])
for r in R:
    t = tr[r.get("trk") or "fixed"]
    t[0] += 1
    t[1] += r["stage"] == "violation"
    t[2] += r["s"]
    c = cfg[tuple(r["a"])]
    c[0] += 1
    c[1] += r["stage"] == "violation"
    c[2] += r["s"]
n, v, s = len(R), sum(t[1] for t in tr.values()), sum(t[2] for t in tr.values())
print(f"{n} tc, {v} violations, {3600 * v / s:.1f} V/h over test-case time "
      f"({S.get('v_per_h', float('nan'))} V/h wall); policy overhead "
      f"{sum(r['ovh_ms'] for r in R) / max(n, 1):.3f} ms/tc")
print("track      tc  share      V    V/h   (95% CI)")
for k, (nt, vt, st) in sorted(tr.items()):
    lo = max(vt - 1.96 * math.sqrt(vt), 0)
    hi = vt + 1.96 * math.sqrt(max(vt, 1))
    print(f"{k:6s} {nt:7d} {nt / n:6.1%} {vt:6d} {3600 * vt / st:6.1f}   "
          f"({3600 * lo / st:.0f}-{3600 * hi / st:.0f})")
inc = tr.get("inc")
if inc and inc[2]:
    print(f"overall / incumbent-track V/h: {(v / s) / (inc[1] / inc[2]):.3f}")
print(f"distinct configurations: {len(cfg)}")
lj = os.path.join(run, "learner.jsonl")
if os.path.exists(lj):
    L = [json.loads(l) for l in open(lj)]
    print(f"learner rounds: {len(L)}, fit seconds median "
          f"{sorted(x['fit_s'] for x in L)[len(L) // 2]}")
    prev = None
    for x in L:
        key = json.dumps(x["incumbent"], sort_keys=True)
        if key != prev:
            print(f"  round {x['round']} ({x['time']}): incumbent {x['incumbent']}")
            prev = key
