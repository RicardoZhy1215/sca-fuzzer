"""
Summary of a same-time A/B run (tcenv.py --policy ctx --ctx-ab gru,tf): per arm, NEW violations
(first per lineage) per 1000 programs and per hour of that arm's own hardware time, overall, by
mode and by block of the run (arms alternate test case by test case, so block k of every arm
covers the same wall-clock window); a conditional binomial test of equal rates; the agents'
training trace.

  python rvzr/SpecRL/genbandit/analyze_ab.py rvzr/SpecRL/genbandit/runs/<run> [--blocks 6]
"""
import argparse
import json
import math
import os

from scipy.stats import binomtest

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--blocks", type=int, default=6)
args = ap.parse_args()


def jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


arms = sorted(d for d in os.listdir(args.run) if os.path.exists(os.path.join(args.run, d, "steps.jsonl")))
S = {a: jsonl(os.path.join(args.run, a, "steps.jsonl")) for a in arms}
A = {a: jsonl(os.path.join(args.run, a, "agent.jsonl")) for a in arms}
H = {a: json.load(open(os.path.join(args.run, a, "agent_header.json")))
     if os.path.exists(os.path.join(args.run, a, "agent_header.json")) else {} for a in arms}


def per1000(rows):
    n, v = len(rows), sum(bool(r["new_v"]) for r in rows)
    if not n:
        return "      -          "
    p = v / n
    se = math.sqrt(max(p * (1 - p), 1e-12) / n)
    return f"{1000 * p:5.1f} ({1000 * max(p - 1.96 * se, 0):4.1f}-{1000 * (p + 1.96 * se):4.1f}) n{n}"


print(f"run {args.run}")
for a in arms:
    h = H[a]
    print(f"  {a}: params {h.get('params')}, tf {h.get('tf')}, cores {h.get('opts', {}).get('cores')}")
print()
print("NEW violations per 1000 programs (95% CI), and per hour of the arm's own hardware time")
for a in arms:
    rows = S[a]
    secs = sum(r["s"] for r in rows)
    v = sum(bool(r["new_v"]) for r in rows)
    dup = sum(bool(r["dup_v"]) for r in rows)
    print(f"  {a:4s} all   {per1000(rows)}   {v} new V, {dup} dup, {v / max(secs, 1) * 3600:6.1f} new V/h "
          f"({secs / 3600:.2f} h of hardware)")
    for m in ("fresh", "rerun", "edit"):
        print(f"       {m:6s}{per1000([r for r in rows if r['mode'] == m])}")

n = min(len(S[a]) for a in arms)
print(f"\nby block (equal program counts per arm, {n} programs each compared)")
print("  block " + "".join(f"{a:>34s}" for a in arms))
for b in range(args.blocks):
    lo, hi = b * n // args.blocks, (b + 1) * n // args.blocks
    print(f"  {b:5d} " + "".join(f"{per1000(S[a][lo:hi]):>34s}" for a in arms))

if len(arms) == 2:
    a, b = arms
    for name, lo in (("whole run", 0), ("second half", n // 2)):
        ra, rb = S[a][lo:n], S[b][lo:n]
        va, vb = sum(bool(r["new_v"]) for r in ra), sum(bool(r["new_v"]) for r in rb)
        if va + vb:
            t = binomtest(vb, va + vb, len(rb) / (len(ra) + len(rb)))
            ratio = (vb / len(rb)) / max(va / len(ra), 1e-12)
            print(f"\n{name}: {b}/{a} = {ratio:.2f}x per program ({vb} vs {va} new V on {len(rb)} vs "
                  f"{len(ra)} programs); conditional binomial p = {t.pvalue:.3g}")

print("\nagent training trace (every ~1/6 of the updates)")
for a in arms:
    tr = A[a]
    if not tr:
        continue
    step = max(1, len(tr) // 6)
    print(f"  {a}: {len(tr)} updates, mean update {sum(x['upd_s'] for x in tr) / len(tr):.2f} s")
    for x in tr[::step] + ([tr[-1]] if (len(tr) - 1) % step else []):
        print(f"    upd {x['update']:4d} h {x['hours']:5.2f}  r_mean {x['r_mean']:.4f}  kl {x['kl']:.3f}  "
              f"ent {x['ent']:.2f}  sil {x['sil']:.2f}  buffer {x['buffer']}")
