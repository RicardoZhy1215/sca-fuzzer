"""
Transfer test summary (run_transfer.sh): per arm, how fast the first violations come and the rate
over the first programs and the whole run; binomial tests between arms at equal programs.

  python rvzr/SpecRL/genbandit/transfer/summarize.py <transfer run dir> [--windows 500,1000,2000]

Violations: RL arms count new ones (first violation of a lineage, as in the reward and the figures);
Revizor counts every violation (it does not rerun or edit programs). Time: RL = the fuzzer's
per-program seconds (steps.jsonl), summed; Revizor = violation directory time minus the arm's start.
"""
import argparse
import glob
import json
import os
import re
import time

from scipy.stats import binomtest

ARMS = [("revizor", "Revizor"), ("cold", "RL from scratch"), ("warm", "RL transferred, learning"),
        ("frozen", "RL transferred, frozen")]


def rl_arm(d):
    t, curve, times = 0.0, [], []
    for line in open(os.path.join(d, "steps.jsonl")):
        r = json.loads(line)
        t += r["s"]
        if r["new_v"]:
            times.append(t)
        curve.append(len(times))
    return curve, times, t


def revizor_arm(d, start):
    text = open(os.path.join(d, "fuzz.log"), errors="replace").read().replace("\r", "\n")
    last = {}
    for n, s in re.findall(r"^(\d+)\s+\(.*Stats: (.*)$", text, re.M):
        last[int(n)] = int(re.search(r"\bV:(\d+)", s).group(1))
    curve, v = [], 0
    for i in range(1, max(last, default=0) + 1):
        v = last.get(i, v)
        curve.append(v)
    vdirs = sorted(glob.glob(os.path.join(d, "violation-*")))
    m = re.search(r"^Duration: ([\d.]+)", text, re.M)
    total = float(m.group(1)) if m else None
    # Revizor prints no progress line for the last program of a -n N run (as in plot_curve.py)
    launch = os.path.join(d, "launch.sh")
    m = re.search(r" -n (\d+) ", open(launch).read()) if os.path.exists(launch) else None
    if m and "Finished at" in text and int(m.group(1)) == len(curve) + 1:
        curve.append(len(vdirs))
    if start is None and total is not None:
        start = os.path.getmtime(os.path.join(d, "fuzz.log")) - total
    times = [time.mktime(time.strptime(os.path.basename(p)[10:], "%y%m%d-%H%M%S")) - start for p in vdirs]
    return curve, times, total


def nth(curve, times, k):
    """(program, minutes) of the k-th violation, or None"""
    if len(times) < k:
        return None
    return next(i + 1 for i, c in enumerate(curve) if c >= k), times[k - 1] / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--windows", default="500,1000,2000")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    starts = {}
    lp = os.path.join(run, "launch.log")
    if os.path.exists(lp):
        for arm, ep in re.findall(r"^(\w+) start (\d{9,})", open(lp).read(), re.M):
            starts[arm] = float(ep)
    res = {}
    for arm, label in ARMS:
        d = os.path.join(run, arm)
        if arm == "revizor" and os.path.exists(os.path.join(d, "fuzz.log")):
            res[arm] = revizor_arm(d, starts.get(arm))
        elif arm != "revizor" and os.path.exists(os.path.join(d, "steps.jsonl")):
            res[arm] = rl_arm(d)
    if not res:
        raise SystemExit(f"no arms in {run}")
    n = min(len(c) for c, _, _ in res.values())
    wins = [w for w in map(int, a.windows.split(",")) if w < n] + [n]
    head = open(os.path.join(run, "machine.txt")).read().splitlines() if os.path.exists(
        os.path.join(run, "machine.txt")) else []
    print(f"# {os.path.basename(run)}")
    for line in head[:2] + [x for x in head if x.startswith(("policy", "spec_store"))]:
        print(f"# {line}")
    print(f"# compared at equal programs: {n}\n")
    cols = ["arm", "programs", "violations", "1st V (program, min)", "10th V (program, min)"] + \
        [f"V/1k first {w}" for w in wins] + ["hours", "V/h"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for arm, label in ARMS:
        if arm not in res:
            continue
        curve, times, total = res[arm]
        f1, f10 = nth(curve, times, 1), nth(curve, times, 10)
        h = total / 3600 if total else None
        row = [label, len(curve), curve[-1] if curve else 0,
               f"{f1[0]}, {f1[1]:.1f}" if f1 else "-", f"{f10[0]}, {f10[1]:.1f}" if f10 else "-"] + \
            [f"{curve[w - 1] / w * 1000:.1f}" for w in wins] + \
            [f"{h:.2f}" if h else "-", f"{curve[-1] / h:.1f}" if h else "-"]
        print("| " + " | ".join(map(str, row)) + " |")
    print("\nbinomial tests (two-sided, split of the two arms' violations at equal programs):")
    for x, y in (("warm", "cold"), ("frozen", "cold"), ("warm", "revizor"), ("frozen", "revizor"),
                 ("cold", "revizor"), ("warm", "frozen")):
        if x not in res or y not in res:
            continue
        parts = []
        for w in wins:
            kx, ky = res[x][0][w - 1], res[y][0][w - 1]
            p = binomtest(kx, kx + ky, 0.5).pvalue if kx + ky else 1.0
            parts.append(f"first {w}: {kx} vs {ky} ({kx / ky:.2f}x, p {p:.2g})" if ky else
                         f"first {w}: {kx} vs {ky} (p {p:.2g})")
        print(f"  {x} vs {y}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
