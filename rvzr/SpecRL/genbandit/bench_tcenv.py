"""
Acceptance test for step 1: does the per-test-case environment cost throughput, or does a
training process on other cores add measurement noise? Alternates four arms on the same (base)
action, --rounds times:
  revizor - unmodified `revizor.py fuzz` (driver.run_config)
  plain   - tcenv.py --mode plain (the re-implemented loop alone)
  bypass  - tcenv.py --mode bypass (features + MLP inference + per-test-case log)
  load    - bypass while a dummy PPO-sized training loop (load_train) runs on cores 4-7
Run it pinned away from the executor's core 0 and the load's cores:
  cd ~/sca-fuzzer
  taskset -c 1-3 python rvzr/SpecRL/genbandit/bench_tcenv.py --minutes 8 --rounds 2 --out <dir>
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, HERE)
import space_v4 as space  # noqa: E402
from driver import PY, executor_busy, run_config  # noqa: E402
from tcenv import best_action  # noqa: E402

# letters of Revizor's brief stats line -> tcenv stage names
LETTERS = {"SF": "spec_filter", "OF": "observ_filter", "Fst": "fast_path", "CN": "fp_nesting",
           "CT": "fp_taint_mistakes", "P1": "fp_early_priming", "CS": "fp_large_sample",
           "P2": "fp_priming", "V": "violation"}

LOAD = """
import torch
torch.set_num_threads(4)
net = torch.nn.Sequential(torch.nn.Linear(82, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                          torch.nn.ReLU(), torch.nn.Linear(256, 32))
opt = torch.optim.Adam(net.parameters(), 1e-3)
x, y = torch.randn(8192, 82), torch.randn(8192, 32)
while True:
    for i in range(0, 8192, 512):
        loss = ((net(x[i:i + 512]) - y[i:i + 512]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
"""


def arm_revizor(action, out, seconds):
    res = run_config(space.write(action), space.V4_DIR, space.SPEC, out, seconds,
                     action["n_inputs"])
    stats = dict(kv.split(":") for kv in res["stats"].split(","))
    stages = {LETTERS[k]: int(v) for k, v in stats.items() if k in LETTERS}
    h = res["fuzz_seconds"] / 3600
    return {"mode": "revizor", "test_cases": res["test_cases"], "seconds": res["fuzz_seconds"],
            "tc_per_min": round(res["test_cases"] / h / 60, 1),
            "violations": len(res["violations"]),
            "v_per_h": round(len(res["violations"]) / h, 1), "stages": stages}


def arm_tcenv(mode, action, out, seconds):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run.log"), "w") as f:
        subprocess.run([PY, os.path.join(HERE, "tcenv.py"), "--mode", mode, "--minutes",
                        str(seconds / 60), "--out", out, "--action", json.dumps(action)],
                       cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return json.load(open(os.path.join(out, "summary.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="revizor,plain,bypass,load")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    action, seconds = best_action(), int(args.minutes * 60)
    for r in range(args.rounds):
        for arm in args.arms.split(","):
            if executor_busy():
                print("[bench] executor busy -- stopping", flush=True)
                return
            out = os.path.join(args.out, f"r{r}-{arm}")
            load = None
            if arm == "load":
                load = subprocess.Popen(["taskset", "-c", "4-7", PY, "-c", LOAD],
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                time.sleep(5)
            try:
                res = arm_revizor(action, out, seconds) if arm == "revizor" else \
                    arm_tcenv("bypass" if arm == "load" else arm, action, out, seconds)
            finally:
                if load:
                    load.kill()
                    load.wait()
            res.update({"round": r, "arm": arm})
            print(json.dumps(res), flush=True)
            with open(os.path.join(args.out, "bench.jsonl"), "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
