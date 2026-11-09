"""
Violations vs number of programs, from unmodified `revizor.py fuzz` logs (one stats line per test
case; V = violations so far). Several run directories are joined in the given order, e.g. a run
cut short and the run that finished its remaining programs. A run's programs are independent of
each other, so the joined curve is that of one run of the total length.

  python rvzr/SpecRL/genbandit/spectre_v1/revizor_curve.py <run dir> [<run dir> ...] [--every 1000]
"""
import argparse
import glob
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--every", type=int, default=1000)
args = ap.parse_args()


def stats(run):
    text = open(os.path.join(run, "fuzz.log"), errors="replace").read().replace("\r", "\n")
    rows = [(int(n), dict(kv.split(":", 1) for kv in s.split(">")[0].strip().split(",") if ":" in kv))
            for n, s in re.findall(r"^(\d+)\s+\(.*Stats: (.*)$", text, re.M)]
    if not rows:
        raise SystemExit(f"{run}: no stats lines yet")
    return rows


curve = {}                     # programs so far (all runs) -> violations so far (all runs)
off_n = off_v = 0
ends = []
tot = {"SF": 0, "OF": 0, "Fst": 0}
for run in args.runs:
    rows = stats(run)
    n, last = rows[-1]
    print(f"{run}: programs {n}, violations {last['V']}, artifacts "
          f"{len(glob.glob(os.path.join(run, 'violation-*')))}"
          + (" (cut short, see ABORTED.txt)" if os.path.exists(os.path.join(run, "ABORTED.txt")) else ""))
    for k, d in rows:
        curve[off_n + k] = off_v + int(d["V"])
    off_n, off_v = off_n + n, off_v + int(last["V"])
    ends.append(off_n)
    for k in tot:
        tot[k] += int(last.get(k, 0))
print(f"\ntotal: programs {off_n}, violations {off_v} ({1000 * off_v / off_n:.1f} per 1000 programs)")
print(f"filters: SF {tot['SF']} (speculation filter), OF {tot['OF']} (observation filter), "
      f"Fst {tot['Fst']} (fast path)")
if len(ends) > 1:
    print("runs joined after programs " + ", ".join(str(x) for x in ends[:-1]))
print(f"\n{'programs':>9s} {'violations':>11s}")
for k in list(range(args.every, off_n + 1, args.every)) + ([off_n] if off_n % args.every else []):
    print(f"{k:9d} {max(v for n, v in curve.items() if n <= k):11d}")
