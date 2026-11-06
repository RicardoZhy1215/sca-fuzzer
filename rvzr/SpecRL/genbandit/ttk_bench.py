"""
Time-to-k benchmark: exact wall-clock time from launching a fuzzing process to its k-th
violation, for Revizor and the RL methods, all on the Spectre v4 campaign. Every run starts a
fresh process; the clock starts right before the launch (so startup -- Python, executor init,
generator construction, agent spawn -- is included) and each violation's time is the earliest
file mtime inside its artifact directory (written once the violation is confirmed). Arms run
round-robin so that drift spreads evenly; every number reported is a measured time, nothing is
extrapolated: a k not reached within the cap is reported as "not reached".

Arms
  revizor_default  unmodified revizor.py fuzz, v4.yaml knobs, -i 50 (the bandit's control)
  revizor_tuned    unmodified revizor.py fuzz, the tuned config (coupled incumbent), -i 10
  progrl_scratch   tcenv --policy progrl, fresh policy (= Revizor's distribution at start)
  ctx_scratch      tcenv --policy ctx, rule modes + lineage reward, fresh policy
  ctx_warm         same, policy warm-started from runs/ctx-rule2-v4-261106-0634/policy.pt

  python rvzr/SpecRL/genbandit/ttk_bench.py --out rvzr/SpecRL/genbandit/runs/ttk-<name> [--rounds 3]
  python rvzr/SpecRL/genbandit/ttk_bench.py --out <same> --summary
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, HERE)
import space_v4 as space  # noqa: E402
from driver import PY, executor_busy, run_config  # noqa: E402

TUNED = json.load(open(os.path.join(HERE, "runs", "progrl-speed-261105", "action.json")))
WARM = os.path.join(HERE, "runs", "ctx-rule2-v4-261106-0634", "policy.pt")
KS = (1, 5, 10, 20, 50, 100)


def violation_times(out, t0):
    """Seconds from launch to each violation: the artifact directory's own mtime (set when its
    files are written, right after the violation is confirmed). NOT the files' mtimes: some are
    copies that keep their source's old mtime (the config yaml), which gave negative times.
    Cross-checked against the second-resolution timestamp in the directory name."""
    import datetime
    ts = []
    for d in glob.glob(os.path.join(out, "violation-*")):
        m = os.stat(d).st_mtime
        named = datetime.datetime.strptime(os.path.basename(d)[len("violation-"):],
                                           "%y%m%d-%H%M%S").timestamp()
        assert -1 <= m - named < 60, f"{d}: dir mtime {m} vs name {named}"
        ts.append(m - t0)
    return sorted(ts)


def run_arm(arm, out, minutes):
    os.makedirs(out, exist_ok=True)
    sec = int(minutes * 60)
    if arm.startswith("revizor"):
        a = space.CONTROL if arm == "revizor_default" else TUNED
        cfg = space.write(a)
        t0 = time.time()
        res = run_config(cfg, space.V4_DIR, space.SPEC, out, sec, a["n_inputs"])
        wall = time.time() - t0
        extra = {"test_cases": res["test_cases"], "rc": res["rc"]}
    else:
        cmd = [PY, os.path.join(HERE, "tcenv.py"), "--mode", "bypass", "--minutes", str(minutes),
               "--out", out, "--action", json.dumps(TUNED)]
        if arm == "progrl_scratch":
            cmd += ["--policy", "progrl"]
        else:
            cmd += ["--policy", "ctx", "--ctx-policy", "learn", "--ctx-mode", "rule2"]
            if arm == "ctx_warm":
                cmd += ["--ctx-init", WARM]
        t0 = time.time()
        with open(os.path.join(out, "run.log"), "w") as f:
            rc = subprocess.run(["taskset", "-c", "1-3"] + cmd, cwd=ROOT, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        wall = time.time() - t0
        s = os.path.join(out, "summary.json")
        extra = {"rc": rc, "test_cases": json.load(open(s))["test_cases"] if os.path.exists(s) else None}
    ts = violation_times(out, t0)
    return {"arm": arm, "out": out, "t0": t0, "wall_s": round(wall, 1), "cap_min": minutes,
            "violations": len(ts), "times_s": [round(x, 2) for x in ts], **extra}


def summary(out):
    R = [json.loads(l) for l in open(os.path.join(out, "ttk.jsonl"))]
    arms = []
    for r in R:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    print("minutes from process launch to the k-th violation (each round = one fresh run); "
          "'-' = not reached within the cap")
    print(f"{'arm':16s} {'round':>5s} {'cap':>5s} " + " ".join(f"{'k=' + str(k):>7s}" for k in KS) + "  violations")
    for a in arms:
        rows = [r for r in R if r["arm"] == a]
        for i, r in enumerate(rows):
            t = violation_times(r["out"] if os.path.isabs(r["out"]) else os.path.join(ROOT, r["out"]),
                                r["t0"])                 # recomputed from the artifacts
            cells = [f"{t[k - 1] / 60:7.2f}" if len(t) >= k else f"{'-':>7s}" for k in KS]
            print(f"{a:16s} {i + 1:5d} {r['cap_min']:5.0f} " + " ".join(cells) + f"  {r['violations']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cap", type=float, default=15.0, help="minutes per run (tuned arms)")
    ap.add_argument("--cap-default", type=float, default=50.0, help="minutes per revizor_default run")
    ap.add_argument("--arms", default="revizor_tuned,progrl_scratch,ctx_scratch,ctx_warm,revizor_default")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(args.out)
        return
    os.makedirs(args.out, exist_ok=True)
    for r in range(args.rounds):
        for arm in args.arms.split(","):
            if executor_busy():
                print("[ttk] executor busy -- stopping", flush=True)
                return
            cap = args.cap_default if arm == "revizor_default" else args.cap
            res = run_arm(arm, os.path.join(args.out, f"r{r}-{arm}"), cap)
            res["round"] = r
            with open(os.path.join(args.out, "ttk.jsonl"), "a") as f:
                f.write(json.dumps(res) + "\n")
            t = res["times_s"]
            print(f"[ttk] round {r} {arm}: {res['violations']} V in {res['wall_s'] / 60:.1f} min; "
                  f"10th at {t[9] / 60:.2f} min" if len(t) >= 10 else
                  f"[ttk] round {r} {arm}: {res['violations']} V in {res['wall_s'] / 60:.1f} min; "
                  f"10th not reached", flush=True)


if __name__ == "__main__":
    main()
