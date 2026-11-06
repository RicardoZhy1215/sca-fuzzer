"""
Edit pilot (tcenv.py --policy edit): is acting on the hardware result worth it?
For each base category (outcome of the fresh program), violation rate of follow-ups by kind:
rerun (same program, new inputs = control), edit_ins, edit_op; vs fresh programs. Also: for
near-miss bases, does the base's observation (excess ops, fenced-divergent inputs) predict
which follow-ups violate; and for violating bases, how often a single edit keeps the violation.

  python rvzr/SpecRL/genbandit/analyze_edit.py rvzr/SpecRL/genbandit/runs/<run>
"""
import json
import math
import os
import sys
from collections import defaultdict

run = sys.argv[1]
R = []
for l in open(os.path.join(run, "edits.jsonl")):
    try:
        r = json.loads(l)
    except json.JSONDecodeError:
        continue
    if r["kind"] in ("mem", "reg"):          # pilot v1 logged the slot kind over "edit_op"
        r["slot_kind"], r["kind"] = r["kind"], "edit_op"
    R.append(r)


def rate(rows):
    n = len(rows)
    v = sum(r["stage"] == "violation" for r in rows)
    if not n:
        return "      -"
    p = v / n
    lo = max(0.0, p - 1.96 * math.sqrt(p * (1 - p) / n)) if v else 0.0
    hi = p + 1.96 * math.sqrt(max(p * (1 - p), 1 / n) / n)
    return f"{n:6d} {v:5d}  {1000 * p:6.1f} ({1000 * lo:5.1f}-{1000 * hi:5.1f})"


fresh = [r for r in R if r["kind"] == "fresh"]
t = sum(r["s"] for r in R)
print(f"{len(R)} test cases, {len(fresh)} episodes, {sum(r['stage'] == 'violation' for r in R)} "
      f"violations, {t / 3600:.2f} h of test-case time")
print("                          n     V   V/1000 (95% CI)")
print(f"fresh programs        {rate(fresh)}")
cats = {"spec": "spec-filtered base", "obs": "obs-filtered base", "near": "near-miss base",
        "viol": "violating base"}
base_of = {r["ep"]: r for r in fresh}
for c, name in cats.items():
    nb = sum(1 for r in fresh if r["cat"] == c)
    print(f"[{name}: {nb} bases]")
    for k in ("rerun", "edit_ins", "edit_op"):
        rows = [r for r in R if r["cat"] == c and r["kind"] == k]
        print(f"  {k:10s}          {rate(rows)}")
# near-miss bases: does the base's observation predict follow-up success?
print("[near-miss bases split by the base's observation]")
for key, cuts in (("ex_max", (0, 60, 120, 10 ** 9)), ("fdiv", (0, 1, 3, 100))):
    for a, b in zip(cuts[:-1], cuts[1:]):
        eps = {e for e, r in base_of.items() if r["cat"] == "near" and r.get(key) is not None
               and a <= r[key] < b}
        rows = [r for r in R if r["ep"] in eps and r["kind"] != "fresh"]
        print(f"  base {key} in [{a},{b}): {len(eps):4d} bases  follow-ups {rate(rows)}")
# edits of near-miss bases: memory vs register operand edits, instruction class
rows = [r for r in R if r["cat"] == "near" and r["kind"] == "edit_op"]
for sk in ("mem", "reg"):
    print(f"  near-miss edit_op on {sk:3s} slot     {rate([r for r in rows if r.get('slot_kind') == sk])}")
# total violations per hour by what produced them
by = defaultdict(lambda: [0, 0.0])
for r in R:
    by[r["kind"]][0] += r["stage"] == "violation"
    by[r["kind"]][1] += r["s"]
print("V/h by test-case kind: " + ", ".join(f"{k} {3600 * v / s:.0f}" for k, (v, s) in by.items() if s))
