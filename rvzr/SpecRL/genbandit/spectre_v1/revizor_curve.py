"""
Violations vs number of programs, from an unmodified `revizor.py fuzz` log (one stats line per test
case; V = violations so far) and the violation artifacts.

  python rvzr/SpecRL/genbandit/spectre_v1/revizor_curve.py <run dir> [--every 1000]
"""
import argparse
import glob
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--every", type=int, default=1000)
args = ap.parse_args()
text = open(os.path.join(args.run, "fuzz.log"), errors="replace").read().replace("\r", "\n")
rows = [(int(n), dict(kv.split(":", 1) for kv in s.split(">")[0].strip().split(",") if ":" in kv))
        for n, s in re.findall(r"^(\d+)\s+\(.*Stats: (.*)$", text, re.M)]
if not rows:
    raise SystemExit("no stats lines yet")
last_n, last = rows[-1]
print(f"{args.run}\nprograms {last_n}, violations {last['V']} "
      f"({1000 * int(last['V']) / last_n:.1f} per 1000 programs), artifacts "
      f"{len(glob.glob(os.path.join(args.run, 'violation-*')))}")
print(f"filters at the end: SF {last.get('SF')} (speculation filter), OF {last.get('OF')} "
      f"(observation filter), Fst {last.get('Fst')} (fast path)")
print(f"\n{'programs':>9s} {'violations':>11s}")
seen = {}
for n, d in rows:
    seen[n] = int(d["V"])
for k in list(range(args.every, last_n + 1, args.every)) + ([last_n] if last_n % args.every else []):
    v = max(val for n, val in seen.items() if n <= k)
    print(f"{k:9d} {v:11d}")
