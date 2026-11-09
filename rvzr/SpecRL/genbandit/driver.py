"""
Generator bandit: Thompson sampling over Revizor generator configurations (arms.py) instead of
an agent that writes programs one instruction at a time.

One pull = one time slice of an unmodified `revizor.py fuzz` run with that arm's config, so the
generator runs at full Revizor speed. Every `--control-every`-th slice runs the control arm
(arms.CONTROL = standard Revizor over everything), so the bandit's rate can be compared with
plain Revizor under identical conditions.

Reward (default, `--reward violations`): violations that survived Revizor's own filters
(nesting, taint, priming, noise, arch-mismatch) per executor-hour. Raw counts reward whatever
fires most often -- including any model false-positive family -- so watch arms whose rate jumps
far above the rest (the masked-move one is blocked in common.yaml).
`--reward classes`: new signatures (signature.py) per executor-hour, signature time included;
~6 min per signature, at most `--max-sig-per-slice` per slice.

Model: per arm, reward ~ Poisson(rate * hours), rate ~ Gamma(alpha, beta); all posteriors
decay toward the prior by `--discount` per slice (1.0 = stationary).

  cd ~/sca-fuzzer
  python rvzr/SpecRL/genbandit/driver.py --hours 12 --out rvzr/SpecRL/genbandit/runs/<name>
  python rvzr/SpecRL/genbandit/driver.py --summary rvzr/SpecRL/genbandit/runs/<name>
The executor is exclusive: the driver refuses to start a slice while another job uses it.
"""
import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, HERE)
import arms  # noqa: E402
from signature import signature, CheckError, PY, SPEC  # noqa: E402

# any python process running Revizor / SpecRL training on the executor (not tied to one machine's
# interpreter path)
BUSY_PATTERN = r"^[^ ]*python[^ ]* ([^ ]*/)?(revizor\.py|train\.py|dr_fp/)"


def executor_busy() -> bool:
    out = subprocess.run(["pgrep", "-f", BUSY_PATTERN], capture_output=True, text=True).stdout
    return bool(out.strip())


def run_slice(arm: str, slice_dir: str, seconds: int, n_inputs: int) -> dict:
    return run_config(os.path.join(arms.ARMS_DIR, f"{arm}.yaml"), HERE, SPEC, slice_dir,
                      seconds, n_inputs)


def run_config(config: str, include_dir: str, spec: str, slice_dir: str, seconds: int,
               n_inputs: int) -> dict:
    """One time slice of an unmodified `revizor.py fuzz` with the given config."""
    os.makedirs(slice_dir, exist_ok=True)
    log = os.path.join(slice_dir, "fuzz.log")
    cmd = [PY, "revizor.py", "fuzz", "-s", spec, "-c", config,
           "-I", include_dir, "-n", "1000000", "-i", str(n_inputs), "-w", slice_dir, "--nonstop",
           "--timeout", str(seconds)]
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    text = open(log, errors="replace").read().replace("\r", "\n")
    stats = re.findall(r"^(\d+)\s+\(.*Stats: (.*)$", text, re.M)
    return {"rc": rc, "fuzz_seconds": round(time.time() - t0, 1),
            "test_cases": int(stats[-1][0]) if stats else 0,
            "stats": stats[-1][1].split(">")[0].strip() if stats else "",
            "violations": sorted(glob.glob(os.path.join(slice_dir, "violation-*")))}


class Bandit:
    def __init__(self, names, alpha0, beta0, discount, rng):
        self.a0, self.b0, self.discount, self.rng = alpha0, beta0, discount, rng
        self.post = {n: [alpha0, beta0] for n in names}

    def choose(self) -> str:
        draws = {n: self.rng.gammavariate(a, 1.0 / b) for n, (a, b) in self.post.items()}
        return max(draws, key=draws.get)

    def update(self, arm: str, reward: float, hours: float) -> None:
        for p in self.post.values():
            p[0] = self.a0 + self.discount * (p[0] - self.a0)
            p[1] = self.b0 + self.discount * (p[1] - self.b0)
        self.post[arm][0] += reward
        self.post[arm][1] += hours


def replay(out: str):
    """Rebuild archives and slice records from slices.jsonl (resume / summary)."""
    path = os.path.join(out, "slices.jsonl")
    recs = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    archives = {"bandit": {}, "control": {}}
    for r in recs:
        for s in r["new_signatures"]:
            archives[r["track"]].setdefault(s, r["slice"])
    return recs, archives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=False)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--slice-min", type=float, default=10.0)
    ap.add_argument("--control-every", type=int, default=6)
    ap.add_argument("--inputs", type=int, default=50)
    ap.add_argument("--reward", choices=["violations", "classes"], default="violations")
    ap.add_argument("--max-sig-per-slice", type=int, default=1)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--beta0", type=float, default=0.5, help="prior hours (mean rate alpha0/beta0)")
    ap.add_argument("--discount", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--summary", metavar="OUT")
    args = ap.parse_args()

    if args.summary:
        summary(args.summary)
        return
    assert args.out, "--out is required"
    os.makedirs(args.out, exist_ok=True)
    names = [n for n in arms.arm_names()]
    rng = random.Random(args.seed)
    bandit = Bandit(names, args.alpha0, args.beta0, args.discount, rng)
    recs, archives = replay(args.out)
    for r in recs:                      # resume: replay the posterior updates
        if r["track"] == "bandit":
            bandit.update(r["arm"], r["reward"], r["hours"])
    spent = sum(r["hours"] for r in recs)
    k = len(recs)
    print(f"[genbandit] {len(names)} arms, control={arms.CONTROL}, reward={args.reward}, "
          f"resume at slice {k}, {spent:.2f} h spent of {args.hours}", flush=True)

    while spent < args.hours:
        if executor_busy():
            print("[genbandit] executor busy -- stopping (another job is using it)", flush=True)
            break
        track = "control" if args.control_every and k % args.control_every == args.control_every - 1 \
            else "bandit"
        arm = arms.CONTROL if track == "control" else bandit.choose()
        sdir = os.path.join(args.out, f"slice-{k:04d}-{arm}")
        res = run_slice(arm, sdir, int(args.slice_min * 60), args.inputs)

        sigs, sig_seconds, unreproducible, skipped, errors = [], 0.0, 0, 0, 0
        for v in (res["violations"] if args.reward == "classes" else []):
            if len(sigs) + unreproducible + errors >= args.max_sig_per_slice:
                skipped += 1
                continue
            t_sig = time.time()
            try:
                sig, info = signature(v, args.inputs)
            except CheckError as e:
                errors += 1
                sig_seconds += time.time() - t_sig
                json.dump({"error": str(e)}, open(os.path.join(v, "signature.json"), "w"))
                continue
            sig_seconds += info["seconds"]
            json.dump({"signature": sorted(sig) if sig is not None else None, **info},
                      open(os.path.join(v, "signature.json"), "w"))
            if sig is None:
                unreproducible += 1
            else:
                sigs.append("+".join(sorted(sig)) or "<none-single-necessary>")
        archive = archives[track]
        new = []
        for s in sigs:
            if s not in archive:
                archive[s] = k
                new.append(s)
        hours = (res["fuzz_seconds"] + sig_seconds) / 3600
        reward = len(new) if args.reward == "classes" else len(res["violations"])
        if track == "bandit":
            bandit.update(arm, reward, hours)
        rec = {"slice": k, "track": track, "arm": arm, "hours": round(hours, 4), "reward": reward,
               "fuzz_seconds": res["fuzz_seconds"], "sig_seconds": round(sig_seconds, 1),
               "test_cases": res["test_cases"], "n_violations": len(res["violations"]),
               "signatures": sigs, "new_signatures": new, "unreproducible": unreproducible,
               "unsigned": skipped, "sig_errors": errors, "rc": res["rc"], "stats": res["stats"],
               "posterior": {a: [round(x, 3) for x in p] for a, p in bandit.post.items()},
               "time": time.strftime("%F %T")}
        with open(os.path.join(args.out, "slices.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[genbandit] slice {k} {track:7s} {arm:16s} tc={res['test_cases']:5d} "
              f"V={len(res['violations'])} reward={reward} ({hours * 60:.1f} min)", flush=True)
        spent += hours
        k += 1
    summary(args.out)


def summary(out: str):
    recs, archives = replay(out)
    if not recs:
        print("no slices yet")
        return
    print(f"== {out}: {len(recs)} slices, {sum(r['hours'] for r in recs):.2f} h")
    for track in ("bandit", "control"):
        rs = [r for r in recs if r["track"] == track]
        h = sum(r["hours"] for r in rs)
        v = sum(r["n_violations"] for r in rs)
        tc = sum(r["test_cases"] for r in rs)
        print(f"  {track:7s}: {len(rs):3d} slices {h:6.2f} h  test cases {tc:7d} "
              f"({tc / h / 60 if h else 0:5.1f}/min)  violations {v:4d} ({v / h if h else 0:5.2f}/h)"
              f"  distinct classes {len(archives[track]):3d}")
    per = {}
    for r in recs:
        p = per.setdefault((r["track"], r["arm"]), [0, 0.0, 0, 0])
        p[0] += 1; p[1] += r["hours"]; p[2] += r["n_violations"]; p[3] += r["test_cases"]
    print("  track    arm               pulls  hours  violations   V/h   tc/min")
    for (t, a), (n, h, v, tc) in sorted(per.items(), key=lambda x: -x[1][1]):
        print(f"  {t:7s}  {a:16s}  {n:5d}  {h:5.2f}  {v:10d}  {v / h if h else 0:5.2f}  "
              f"{tc / h / 60 if h else 0:6.1f}")
    for track in ("bandit", "control"):
        if archives[track]:
            print(f"  {track} classes:")
            for s, k in archives[track].items():
                print(f"    slice {k:4d}: {s}")


if __name__ == "__main__":
    main()
