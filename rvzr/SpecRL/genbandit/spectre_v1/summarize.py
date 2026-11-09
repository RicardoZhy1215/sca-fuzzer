"""
One row per Revizor run in spectre_v1/runs: programs, violations, violations per 1000 programs,
filter counts, run status, and how its config differs from a reference yaml (default configs/v1.yaml).

  python rvzr/SpecRL/genbandit/spectre_v1/summarize.py [--ref configs/v1.yaml]
"""
import argparse
import glob
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--ref", default=os.path.join(HERE, "configs", "v1.yaml"))
args = ap.parse_args()
ref = yaml.safe_load(open(args.ref))


def diff(c):
    out = []
    for k in sorted(set(ref) | set(c)):
        if ref.get(k) != c.get(k):
            v = c.get(k, "-")
            if isinstance(v, list):
                a, b = set(map(str, v)), set(map(str, ref.get(k) or []))
                v = " ".join([f"+{x}" for x in sorted(a - b)] + [f"-{x}" for x in sorted(b - a)])
            out.append(f"{k}={v}")
    return ", ".join(out) or "(same as ref)"


print(f"ref: {args.ref}")
print(f"{'run':44s} {'status':8s} {'programs':>8s} {'V':>5s} {'V/1000':>7s} {'SF':>6s} {'OF':>6s} {'Fst':>6s}  config vs ref")
for r in sorted(glob.glob(os.path.join(HERE, "runs", "revizor-*"))):
    log = os.path.join(r, "fuzz.log")
    if not os.path.exists(log):
        status, rows = "queued", []
    else:
        text = open(log, errors="replace").read().replace("\r", "\n")
        rows = re.findall(r"^(\d+)\s+\(.*Stats: (.*)$", text, re.M)
        q = open(os.path.join(r, "queue.log")).read() if os.path.exists(os.path.join(r, "queue.log")) else ""
        status = "aborted" if os.path.exists(os.path.join(r, "ABORTED.txt")) else \
            "done" if "end " in q else "running"
    cfg = yaml.safe_load(open(os.path.join(r, "config.yaml")))
    if rows:
        n, s = rows[-1]
        d = dict(kv.split(":", 1) for kv in s.split(">")[0].strip().split(",") if ":" in kv)
        n, v = int(n), int(d["V"])
        print(f"{os.path.basename(r):44s} {status:8s} {n:8d} {v:5d} {1000 * v / n:7.1f} "
              f"{d['SF']:>6s} {d['OF']:>6s} {d['Fst']:>6s}  {diff(cfg)}")
    else:
        print(f"{os.path.basename(r):44s} {status:8s} {'-':>8s} {'-':>5s} {'-':>7s} {'':6s} {'':6s} {'':6s}  {diff(cfg)}")
