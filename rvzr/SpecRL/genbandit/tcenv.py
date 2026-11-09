"""
Per-test-case environment around Revizor's fuzzing loop (step 1 of the DNN plan).

Runs the loop of Fuzzer.start one test case at a time, with hooks where a policy can act and
observe:
  1. before generation - the policy picks the program-side config of this test case
  2. after generation  - static program features (features.py); the policy picks the input side
  3. after the filters and the fuzzing round - the stage that ended the test case, PFC readings
Every test case is logged to tc.jsonl (stage, seconds, policy overhead, PFCs, features).

Modes:
  plain  - the loop only: no features, no inference, no capture, no per-test-case log
  bypass - features + one forward pass per hook of a 2x256 numpy MLP whose last layer is zero
           (a residual on the base action), but the action is always the base action:
           identical behavior, real overhead

Every knob can change per test case: the fuzzer is built from the superset config (all
optional categories, lock forms, all 6 GPRs) so its asm parser accepts any program, and
programs come from a cache of generators keyed by (categories, locks, GPR pool). With
--perturb P (step 2: data collection) a test case runs the base action with probability 1 - P,
else the base action with one uniformly chosen knob set to another uniformly chosen value.

With --policy coupled the configuration of every test case comes from coupled.py (knobs chosen
jointly from a model with pairwise interactions); its learner runs as a separate process on
cores 4-7 and exchanges policy.json / tc.jsonl with this loop. All 48 generators are built
up front.

  cd ~/sca-fuzzer
  python rvzr/SpecRL/genbandit/tcenv.py --mode bypass --minutes 10 --out rvzr/SpecRL/genbandit/runs/tc-<name>
  python rvzr/SpecRL/genbandit/tcenv.py --mode bypass --perturb 0.1 --minutes 120 --out ...
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import space_v4 as space  # noqa: E402
from features import FeatureExtractor  # noqa: E402

BEST_FROM = os.path.join(HERE, "runs", "v4-rl-261103-0208")
# STAT counters that tell which stage ended a test case, in pipeline order
STAGES = ["spec_filter", "observ_filter", "fast_path", "fp_nesting", "fp_taint_mistakes",
          "fp_priming", "fp_large_sample"]
PROGRAM_KNOBS = ["program_size", "mem_ratio", "n_regs", "binary", "flags", "semaphore", "locks"]
INPUT_KNOBS = ["entropy_bits", "n_inputs", "value_mix", "spec_filter"]
# knobs that change the generator's instruction set, and their most permissive values
GEN_KNOBS = ["binary", "flags", "semaphore", "locks", "n_regs"]
SUPERSET = {"binary": "on", "flags": "on", "semaphore": "on", "locks": "on", "n_regs": 6}


def best_action(run_dir: str = BEST_FROM) -> dict:
    """Most probable value of every knob in the last policy snapshot of a v4 RL run."""
    last = [json.loads(l) for l in open(os.path.join(run_dir, "slices.jsonl"))][-1]["policy"]
    return {k: space.KNOBS[k][max(range(len(p)), key=p.__getitem__)] for k, p in last.items()}


class ResidualMLP:
    """2x256 ReLU MLP; the last layer starts at zero, so a fresh network adds nothing to the
    base policy's logits."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, hidden: int = 256):
        self.w1 = rng.normal(0, 1 / math.sqrt(n_in), (n_in, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.w2 = rng.normal(0, 1 / math.sqrt(hidden), (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, np.float32)
        self.w3 = np.zeros((hidden, n_out), np.float32)
        self.b3 = np.zeros(n_out, np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ self.w1 + self.b1, 0)
        h = np.maximum(h @ self.w2 + self.b2, 0)
        return h @ self.w3 + self.b3


class TCEnv:
    def __init__(self, action: dict, mode: str, out: str, seed: int = 1, perturb: float = 0.0,
                 actor=None):
        from rvzr.config import CONF
        from rvzr.logs import update_logging_after_config_change
        from rvzr.factory import get_fuzzer
        from rvzr.stats import FuzzingStats
        from rvzr.arch.x86.fuzzer import _check_instruction_list
        assert mode == "bypass" or not perturb, "perturbed runs are for logging: use bypass"
        self.CONF, self.STAT = CONF, FuzzingStats()
        self.action, self.mode, self.out = dict(action), mode, out
        self.perturb, self.prng, self.pk = perturb, random.Random(seed), None
        self.actor, self.trk = actor, None
        assert mode == "bypass" or actor is None, "the coupled policy logs to tc.jsonl: use bypass"
        cfg = space.write({**action, **SUPERSET})
        CONF.load(cfg, os.path.dirname(cfg))
        update_logging_after_config_change()
        self.fuzzer = get_fuzzer(space.SPEC, out, "", None)
        _check_instruction_list(self.fuzzer._isa_spec)
        self.gens = {}
        self._generator(action)
        if perturb:   # build every one-knob neighbor now, outside the timed loop
            for k in GEN_KNOBS:
                for v in space.KNOBS[k]:
                    self._generator({**action, k: v})
        if actor and getattr(actor, "needs_all_generators", True):   # any config can come up
            import itertools
            t = time.perf_counter()
            for vals in itertools.product(*[space.KNOBS[k] for k in GEN_KNOBS]):
                self._generator({**action, **dict(zip(GEN_KNOBS, vals))})
            print(f"[tcenv] built {len(self.gens)} generators in {time.perf_counter() - t:.1f} s",
                  flush=True)
        self.captured = []
        if mode == "bypass":
            rng = np.random.default_rng(seed)
            self.fx = FeatureExtractor(space.SPEC)
            n_prog = sum(len(space.KNOBS[k]) for k in PROGRAM_KNOBS) + 1      # + value head
            n_inp = sum(len(space.KNOBS[k]) for k in INPUT_KNOBS) + 2 + 1      # + skip, value
            self.ctx = np.zeros(4, np.float32)  # EMA of violation, seconds, filtered, fast path
            self.pi_program = ResidualMLP(len(self.ctx), n_prog, rng)
            self.pi_inputs = ResidualMLP(len(self.fx.names), n_inp, rng)
            ex = self.fuzzer.executor
            orig = ex.trace_test_case

            def capture(*args, **kwargs):
                r = orig(*args, **kwargs)
                self.captured.append(r)
                return r
            ex.trace_test_case = capture

    def _generator(self, a: dict):
        """Program generator for the action's categories, lock forms and GPR pool, built with
        exactly the instruction set that action's yaml would give."""
        key = tuple(a[k] for k in GEN_KNOBS)
        if key not in self.gens:
            from rvzr.isa_spec import InstructionSet
            from rvzr.factory import get_program_generator
            C = self.CONF
            saved = (C.instruction_categories, C.instruction_blocklist, C.register_blocklist)
            try:
                C.instruction_categories = space.categories(a)
                C.instruction_blocklist = list(saved[1]) + \
                    (space._lock_names() if a["locks"] == "off" else [])
                C.register_blocklist = space.register_blocklist(a)
                isa = InstructionSet(space.SPEC, C.instruction_categories)
                self.gens[key] = get_program_generator(C.program_generator_seed, isa)
            finally:
                C.instruction_categories, C.instruction_blocklist, C.register_blocklist = saved
        return self.gens[key]

    def _apply_program(self, a: dict):
        self.CONF.program_size = int(a["program_size"])
        self.CONF.avg_mem_accesses = max(1, round(a["mem_ratio"] * a["program_size"]))
        return self._generator(a)

    def _apply_inputs(self, a: dict) -> None:
        self.CONF.data_generator_entropy_bits = a["entropy_bits"]
        self.fuzzer.data_gen.max_input_value = pow(2, a["entropy_bits"])
        self.CONF.input_value_mix = a["value_mix"]
        self.CONF.enable_speculation_filter = a["spec_filter"] == "on"

    def _act_program(self) -> dict:
        self.pi_program(self.ctx[None, :])          # residual is zero: the base action
        a, self.pk = dict(self.action), None
        if self.actor:
            a, self.trk = self.actor.choose()
            return a
        if self.perturb and self.prng.random() < self.perturb:
            self.pk = self.prng.choice(list(space.KNOBS))
            a[self.pk] = self.prng.choice([v for v in space.KNOBS[self.pk] if v != a[self.pk]])
        return a

    def _act_inputs(self, feats, a: dict) -> dict:
        self.pi_inputs(np.asarray(feats, np.float32)[None, :])
        return a

    def _info(self, stage: str) -> dict:
        """Hardware observation of the test case just run (bypass mode only: needs the captured
        traces): per input, dispatched - retired ops ("excess"), and how many inputs' traces
        change when the program is fenced (only known past the speculation filter)."""
        org = self.captured[0] if self.captured else None
        if not org:
            return {}
        ex = [h.get_max_pfc()[0] - h.get_max_pfc()[1] for h in org]
        out = {"ex_max": int(max(ex)), "ex_mean": round(sum(ex) / len(ex), 1), "n_in": len(org),
               "ex_pos": int(sum(e > 0 for e in ex))}
        if stage != "spec_filter" and len(self.captured) > 1:
            fen = self.captured[1]
            out["fdiv"] = int(sum(not self.fuzzer.analyser.htraces_are_equivalent(a, b)
                                  for a, b in zip(fen, org)))
        return out

    def run(self, seconds: float, max_programs: int = 0) -> dict:
        f, CONF, STAT = self.fuzzer, self.CONF, self.STAT
        bypass = self.mode == "bypass"
        log = open(os.path.join(self.out, "tc.jsonl"), "w", buffering=1 << 16) if bypass else None
        if bypass:
            json.dump({"action": self.action, "perturb": self.perturb, "knobs": space.KNOBS,
                       "policy": type(self.actor).__name__ if self.actor else "fixed",
                       "features": self.fx.names},
                      open(os.path.join(self.out, "header.json"), "w"))
        stages = {s: 0 for s in STAGES + ["violation", "other"]}
        # whole-iteration seconds per stage (sum, sum of squares): compares modes on equal work
        it_s = {s: [0.0, 0.0] for s in stages}
        ovh_total, n = 0.0, 0
        f.log.start(1000000, datetime.today())
        t_start = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            ovh = 0.0
            if bypass:
                a = self._act_program()
                ovh += time.perf_counter() - t0
            else:
                a = self.action
            f.log.start_round(n)
            gen = self._apply_program(a)
            if self.actor is not None and hasattr(self.actor, "before_generate"):
                th = time.perf_counter()
                self.actor.before_generate(gen)      # progrl: install this program's decisions
                ovh += time.perf_counter() - th
            test_case = gen.create_test_case(f._existing_test_case)
            STAT.test_cases += 1
            feats = None
            if bypass:
                t1 = time.perf_counter()
                feats = self.fx(test_case)
                a = self._act_inputs(feats, a)
                ovh += time.perf_counter() - t1
                pre = time.perf_counter() - t0   # spent before anything runs: what a skip saves
            self._apply_inputs(a)
            inputs = f.data_gen.generate(a["n_inputs"], n_actors=test_case.n_actors())
            STAT.num_inputs += len(inputs) * CONF.inputs_per_class
            before = [getattr(STAT, s) for s in STAGES]
            self.captured.clear()
            violation = None
            if not f._filter(test_case, inputs):
                pfc_trace = self.captured[0] if self.captured else None
                violation = f.fuzzing_round(test_case, inputs, [])
                if violation:
                    f.log.report_violations(violation)
                    f.log.dbg_violation(violation, f.model)
                    f._store_violation_artifact(violation, f._work_dir)
                    STAT.violations += 1
            else:
                pfc_trace = self.captured[0] if self.captured else None
            changed = [s for s, b in zip(STAGES, before) if getattr(STAT, s) != b]
            stage = "violation" if violation else (changed[0] if changed else "other")
            stages[stage] += 1
            t_end = time.perf_counter()
            if self.actor is not None and hasattr(self.actor, "observe"):
                if getattr(self.actor, "wants_info", False):
                    self.actor.observe(stage, t_end - t0, self._info(stage))
                else:
                    self.actor.observe(stage, t_end - t0)
            if bypass:
                t2 = time.perf_counter()
                pfc = [max(h.get_max_pfc()[j] for h in pfc_trace) for j in range(5)] \
                    if pfc_trace else None
                log.write(json.dumps({"i": n, "s": round(t_end - t0, 4), "ovh_ms": round(ovh * 1000, 3),
                                      "pre_ms": round(pre * 1000, 3), "stage": stage, "pfc": pfc, "pk": self.pk, "trk": self.trk,
                                      "a": [space.KNOBS[k].index(a[k]) for k in space.KNOBS],
                                      "x": feats,
                                      **(self.actor.log_extra() if hasattr(self.actor, "log_extra") else {})})
                          + "\n")
                v, dt = float(stage == "violation"), t_end - t0
                self.ctx = 0.99 * self.ctx + 0.01 * np.array(
                    [v, dt, stage in ("spec_filter", "observ_filter"), stage == "fast_path"],
                    np.float32)
                ovh += time.perf_counter() - t2
            ovh_total += ovh
            it = time.perf_counter() - t0
            it_s[stage][0] += it
            it_s[stage][1] += it * it
            n += 1
            if t_end - t_start > seconds or (max_programs and n >= max_programs):
                f.log.timeout()
                break
        f.log.finish()
        if log:
            log.close()
        wall = time.perf_counter() - t_start
        res = {"mode": self.mode, "test_cases": n, "seconds": round(wall, 1),
               "tc_per_min": round(n / wall * 60, 1), "violations": stages["violation"],
               "v_per_h": round(stages["violation"] / wall * 3600, 1),
               "overhead_ms_per_tc": round(ovh_total / n * 1000, 3) if n else 0,
               "overhead_frac": round(ovh_total / wall, 5), "stages": stages,
               "iteration_s": {s: {"n": stages[s], "sum": round(a, 4), "sumsq": round(b, 6)}
                               for s, (a, b) in it_s.items() if stages[s]},
               "action": self.action, "time": time.strftime("%F %T")}
        json.dump(res, open(os.path.join(self.out, "summary.json"), "w"), indent=1)
        return res


class CoupledActor:
    """Per-test-case side of the coupled policy: incumbent with prob 1 - alpha, else the
    learner's candidate (share cand_frac) or one random bootstrap member's best configuration.
    Re-reads policy.json (written atomically by coupled.py learn) every 50 test cases."""

    def __init__(self, out: str, incumbent: dict, seed: int = 1):
        self.path, self.inc, self.rng = os.path.join(out, "policy.json"), dict(incumbent), random.Random(seed)
        self.cand, self.explore, self.alpha, self.cand_frac, self.mtime, self.n = None, [], 0.0, 0.5, None, 0

    def _reload(self):
        try:
            m = os.stat(self.path).st_mtime_ns
        except FileNotFoundError:
            return
        if m != self.mtime:
            p = json.load(open(self.path))
            self.mtime, self.inc, self.cand, self.explore = m, p["incumbent"], p["candidate"], p["explore"]
            self.alpha, self.cand_frac = p["alpha"], p.get("cand_frac", 0.5)

    def choose(self):
        if self.n % 50 == 0:
            self._reload()
        self.n += 1
        if self.cand and self.rng.random() < self.alpha:
            if self.rng.random() < self.cand_frac:
                return dict(self.cand), "cand"
            return dict(self.rng.choice(self.explore)), "exp"
        return dict(self.inc), "inc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=["v4", "v1"], default="v4",
                    help="config space: v4 = store bypass (space_v4), v1 = bounds-check bypass with "
                         "several basic blocks (spectre_v1/space_v1ctx)")
    ap.add_argument("--mode", choices=["plain", "bypass"], default="bypass")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--max-programs", type=int, default=0, help="also stop after this many programs (0: no cap)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--action", help="JSON action dict; default: argmax of the v4 RL run")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="probability that a test case changes one knob of the base action")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy", choices=["fixed", "coupled", "progrl", "edit", "ctx"], default="fixed")
    ap.add_argument("--ctx-policy", choices=["learn", "rule", "fresh"], default="learn",
                    help="ctx: learned context policy, the edit-pilot rule, or fresh programs only")
    ap.add_argument("--ctx-mode", choices=["learn", "rule2"], default="learn",
                    help="ctx learn: modes from the policy head, or the fixed rule (violation/filtered "
                         "-> fresh, near miss -> rerun/edit)")
    ap.add_argument("--ctx-init", default="", help="ctx learn: warm-start policy.pt")
    ap.add_argument("--ctx-E", type=int, default=16, help="ctx: interleaved episodes")
    ap.add_argument("--ctx-N", type=int, default=8, help="ctx: programs per episode")
    ap.add_argument("--ctx-arch", choices=["gru", "tf"], default="gru", help="ctx learn: policy network")
    ap.add_argument("--ctx-ab", default="",
                    help="ctx learn: same-time A/B of comma-separated archs (e.g. gru,tf), one agent "
                         "and out subdir per arm, alternating test cases; learner cores are split")
    ap.add_argument("--tf-layers", type=int, default=2)
    ap.add_argument("--tf-heads", type=int, default=4)
    ap.add_argument("--tf-ff", type=int, default=512)
    ap.add_argument("--rl-lr", type=float, default=3e-4, help="progrl: Adam learning rate")
    ap.add_argument("--rl-kl", type=float, default=0.1, help="progrl: KL-to-Revizor weight")
    ap.add_argument("--rl-sil", type=float, default=1.0, help="progrl: self-imitation weight")
    ap.add_argument("--rl-frozen", action="store_true", help="progrl: never update (prior policy)")
    ap.add_argument("--alpha", type=float, default=0.15, help="coupled: exploration share")
    ap.add_argument("--cand-frac", type=float, default=0.5, help="coupled: candidate share of it")
    ap.add_argument("--learner-cores", default="4-7")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global space
    if args.space == "v1":
        sys.path.insert(0, os.path.join(HERE, "spectre_v1"))
        import space_v1ctx as space
        action = {**space.CONTROL, **json.loads(args.action)} if args.action else dict(space.CONTROL)
    else:
        action = json.loads(args.action) if args.action else best_action()
    os.chdir(ROOT)
    actor, learner = None, None
    if args.policy == "coupled":
        import subprocess
        out = os.path.abspath(args.out)
        for f in ("policy.json", "summary.json"):
            if os.path.exists(os.path.join(out, f)):
                os.remove(os.path.join(out, f))
        actor = CoupledActor(out, action, args.seed)
        learner = subprocess.Popen(
            ["taskset", "-c", args.learner_cores, sys.executable, os.path.join(HERE, "coupled.py"),
             "learn", "--run", out, "--incumbent", json.dumps(action), "--alpha", str(args.alpha),
             "--cand-frac", str(args.cand_frac), "--seed", str(args.seed)],
            stdout=open(os.path.join(out, "learner.log"), "a"), stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"})
    if args.policy == "ctx":
        from ctxrl import CtxActor, CtxABActor
        a, b = (int(x) for x in args.learner_cores.split("-"))
        cores = list(range(a, b + 1))
        tf = {"tf_layers": args.tf_layers, "tf_heads": args.tf_heads, "tf_ff": args.tf_ff}

        def ctx_opts(arch, seed, cores):
            return {"policy": args.ctx_policy, "mode": args.ctx_mode,
                    "init": os.path.abspath(args.ctx_init) if args.ctx_init else "",
                    "E": args.ctx_E, "N": args.ctx_N, "lr": args.rl_lr,
                    "kl": args.rl_kl, "sil": args.rl_sil, "seed": seed, "cores": cores,
                    "threads": len(cores), "arch": arch, "tf": tf if arch == "tf" else None}
        if args.ctx_ab:
            arms = args.ctx_ab.split(",")
            assert not args.ctx_init, "A/B arms start from the same (Revizor) prior"
            k = len(cores) // len(arms)
            actor = CtxABActor(os.path.abspath(args.out), action,
                               [(arm, ctx_opts(arm, args.seed + 1000 * i, cores[i * k:(i + 1) * k]))
                                for i, arm in enumerate(arms)])
        else:
            actor = CtxActor(os.path.abspath(args.out), action, ctx_opts(args.ctx_arch, args.seed, cores))
    if args.policy == "edit":
        from progrl import EditProbeActor
        actor = EditProbeActor(os.path.abspath(args.out), action, {"seed": args.seed})
    if args.policy == "progrl":
        from progrl import ProgActor
        a, b = (int(x) for x in args.learner_cores.split("-"))
        actor = ProgActor(os.path.abspath(args.out), action,
                          {"lr": args.rl_lr, "kl": args.rl_kl, "sil": args.rl_sil,
                           "frozen": args.rl_frozen, "seed": args.seed, "cores": list(range(a, b + 1))})
    try:
        res = TCEnv(action, args.mode, args.out, args.seed, args.perturb, actor).run(args.minutes * 60, args.max_programs)
    except BaseException:
        if learner:
            learner.terminate()
        raise
    finally:
        if actor is not None and hasattr(actor, "close"):
            actor.close()
        if learner and learner.poll() is None:   # it does a last round once summary.json exists, then exits
            try:
                learner.wait(timeout=600)
            except Exception:
                learner.terminate()
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
