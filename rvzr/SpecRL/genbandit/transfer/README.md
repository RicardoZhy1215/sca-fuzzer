# Transfer: a Spectre v4 policy trained on one Intel machine, used on another

Claim under test: a ctxrl policy trained on machine A finds violations on machine B from its first
programs, without the learning phase a from-scratch run needs there. So the learned knowledge is
about the leak (what v4-prone programs look like), not about one CPU.

## The policy (`v4-i7-6700/`)

ctxrl GRU (416k parameters), rule2 modes, trained on an Intel Core i7-6700 (Skylake, microcode 0xf0,
SMT off) for 20000 programs / 3.6 h / 78 PPO updates; 260 new violations (Revizor 131 on the same
programs). Config: v4.yaml knobs + 64 instructions, 1 block, SSBD off, `-i 50` (`config.yaml`, =
`v4/c441d47e7c.yaml`). Source run: `genbandit/runs/cmp-v4-ml2558-...-20000-261112-1419/ctx`
(gitignored, on that machine).

| file | what |
|---|---|
| `policy.pt` | final weights (state dict of the GRU policy, `ctxrl.make_policy`) |
| `table.json` | the instruction table the policy's outputs index; `ctxrl.py` refuses the warm start if the target builds a different one (other base.json, config or generator) |
| `config.yaml` | the Revizor config; `run_transfer.sh` refuses to run if `space_v4` writes a different one |
| `machine.txt`, `header.json`, `agent_header.json`, `summary.json` | source machine, knobs, agent options, run totals |

The learning phase it skips. From scratch on the i7-6700, new violations per 1000 programs, per 2000
programs: 5.5, 10.5, 12.5, 12.5, 17.0, 13.0, 17.5, 12.0, 19.5, 10.0 (Revizor: 6.55 overall). The
first 2000 programs are at Revizor's rate; it reaches its working rate (~13-15/1000) after about
2000-4000 programs (~0.5 h). If the policy transfers, the warm-started arms start at that rate.

## Arms (`run_transfer.sh`, all on the target machine, same config, same number of programs, in turn)

| arm | what | shows |
|---|---|---|
| `revizor` | unmodified Revizor | the baseline on this CPU |
| `cold` | ctxrl from scratch (Revizor's prior), learning here | the learning phase on this CPU |
| `warm` | ctxrl from the transferred policy, still learning | transfer + adaptation |
| `frozen` | the transferred policy, never updated (`--rl-lr 0`) | zero-shot transfer: what the policy itself carries |

Same seed for `cold`, `warm` and `frozen`, so they differ only in the starting weights (and updates).

## Running on the target

```
cd <sca-fuzzer checkout> && git pull            # branch spectre-v1; base.json present
PY=<python with Revizor deps + torch> rvzr/SpecRL/genbandit/transfer/run_transfer.sh 5000 <fuzzer cores> <learner cores>
  e.g. ... run_transfer.sh 5000 1 2-3                # 4-core machine, SMT off
       ARMS="revizor warm" ... run_transfer.sh 5000   # a subset
       POLICY=<other>/policy.pt ...                   # another policy (with its table.json, config.yaml)
```

- Detached; waits for the executor lock and any other fuzzer. NMI watchdog must be 0 (the run
  aborts otherwise); the `rvzr_executor` module built and loaded; on hybrid CPUs P-cores only.
- SSBD must not be forced on by the kernel (`spec_store_bypass` in `machine.txt` should say
  "prctl" or "seccomp", not "Speculative Store Bypass disabled" globally), or no arm sees v4.
- Output: `transfer/runs/v4-<host>-<cpu>-<N>-<date>/{revizor,cold,warm,frozen}/`, `launch.log`,
  `machine.txt`, `RESULTS.txt`; figure `transfer/figs/<same name>.{png,pdf,csv}` (cumulative
  violations vs programs, one line per arm).
- N = 5000: covers the from-scratch learning phase (first 2000-4000 programs); after that the cold
  arm has caught up, so a longer run only dilutes the gap. Time on the i7-6700: Revizor ~35 min,
  each RL arm ~40-60 min (more of its programs reach confirmation), ~3.5 h for all four.

## Reading `RESULTS.txt` (`summarize.py`)

Per arm: programs, violations (RL: new ones = first of a lineage), the program index and fuzzing
minutes of the 1st and 10th violation, violations per 1000 over the first 500 / 1000 / 2000 / all
programs, hours, violations per hour. Then two-sided binomial tests between arms at equal programs.

Transfer holds if `warm` and `frozen` beat `cold` over the first 1000-2000 programs (and reach the 1st
and 10th violation sooner); `frozen` ≈ `warm` there means the gain is carried by the policy, not
re-learned. Compare arms within the target machine only: its absolute v4 rate can differ from the
i7-6700's (another store-forwarding predictor). One seed per arm: if a gap is marginal, rerun.

## Results
