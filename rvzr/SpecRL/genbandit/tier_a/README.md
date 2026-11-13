# Tier A: rare known leaks from the Revizor papers, Revizor vs ctxrl

Why these: ctxrl helps most where violations are rare (SSE FP: 178 vs 8 violations in 20k programs) and
not at all where they are common (Spectre v1 under CT-SEQ: 748 vs 729). The three Revizor papers report
leaks that are rare, have a known answer, and need only a config change (one actor, generator path):

- Hide and Seek with Spectres (Oleksenko et al., S&P'23, arXiv 2301.07642): SCO, V1-Var, ZDI
- Speculation at Fault (Hofmann et al., USENIX Sec'23): exceptions vs the CT-DH / CT-VS contracts
- Enter, Exit, Page Fault, Leak (Oleksenko et al., S&P'25, arXiv 2507.06039): actors; that is tier B,
  not here (see "Next" below)

Every run gives unmodified Revizor and ctxrl (GRU, learned modes) the same config and the same number
of programs, one after the other (not at the same time), and plots cumulative violations vs programs.

## Configs (`configs/`)

| file | leak | contract | what violates it | paper result (rarity) | Zen 4 smoke test (300 programs, Revizor) |
|---|---|---|---|---|---|
| `sco.yaml` | String Comparison Overrun: REPE/REPNE CMPS/SCAS reads past the bound while rcx is unresolved | CT-SEQ | the over-read string bytes | rarest of the paper: 4 per 500k programs (Intel Xeon), ~4 h to the first with all optimizations, never with random inputs; also on AMD Zen 3 | 0 violations, 0.28 s/program |
| `v1var.yaml` | V1-Var | **CT-COND** (plain v1 allowed) | leaks the branch-misprediction model does not explain | 3 per 100k (Xeon), ~75 min; a microarchitectural race; AMD not reported | 0, 0.18 s/program |
| `de-dh.yaml` | divide-by-zero #DE, transient quotient | **CT-DH** (`delayed-exception-handling`) | dependents of the faulting division run with a value (AMD: non-zero) | AMD Zen+/Zen 3: first violation after 312-785 test cases; Intel 312-9k | **1 violation** (test case 226), 0.21 s/program |
| `de-ops.yaml` | Divider State Sampling (DSS) | CT-VS-Ops (`vspec-ops-div`) | quotient depends on an earlier division | Zen+ only, after 80k test cases; Zen 3 clean | **does not run**: `vspec-ops-div` raises NotImplementedError ("no longer maintained") in this Revizor; kept for reference |
| `zdi.yaml` | Zero Dividend Injection: 64-bit DIV with a slow rdx assumes rdx = 0 | CT-SEQ | the injected zero | Intel only: 11 per 500k, ~21 min; Zen 3 has no division speculation | 0; the speculation filter dropped 149/150 programs (ctxrl smoke test): no speculation, as on Zen 3 |

Common settings: random seeds (0), `inputs_per_class: 2` (contract-driven inputs; random inputs find none
of these), both filters on, P+P, SSBD on (`x86_executor_enable_ssbp_patch: true`, so no Spectre v4),
straight-line code except `v1var` (2 blocks x 2 successors, = SpecRL's `spectre_v1/configs/v1.yaml`
with CT-COND). `sco` and `zdi` follow Revizor's `demo/detect-*.yaml`; `de-dh` follows Speculation at
Fault (32 instructions, 8 memory accesses, no CMOV, one exception type per run).

ctxrl smoke tests (150 programs each) ran without errors on all four working configs.

## Running (any x86 machine with the executor kernel module built and loaded)

```
cd <sca-fuzzer checkout>                     # branch spectre-v1, base.json present
PY=<python with Revizor's deps + torch> rvzr/SpecRL/genbandit/tier_a/run_compare.sh <config> <N> [fuzzer cores] [learner cores]
  e.g. rvzr/SpecRL/genbandit/tier_a/run_compare.sh de-dh 10000            # cores 1-3 fuzzer, 4-7 learner
       ARMS=revizor rvzr/SpecRL/genbandit/tier_a/run_compare.sh zdi 20000  # Revizor only
       INLINE=1 ...                                                        # block instead of detaching
```

- Detached by default (setsid nohup); a second run waits on `tier_a/.executor.lock` and for any other
  fuzzer to finish (the executor is one global resource: never run two campaigns at once).
- Output: `tier_a/runs/<config>-<host>-<cpu>-<N>-<date>/` with `config.yaml` (exact config used),
  `machine.txt` (CPU, microcode, kernel, SMT, cmdline, git commit), `revizor/` (violation dirs,
  fuzz.log), `ctx/` (steps.jsonl, tc.jsonl, agent logs, violation dirs), `launch.log`; figure
  `tier_a/figs/<same name>.{png,pdf,csv}`.
- A config not in `configs/`: pass its path instead of a name (`--space yaml` takes any Revizor yaml
  without `register_blocklist`; every knob is fixed to the yaml, the policy only writes programs).
- Intel: give P-cores only on hybrid CPUs (E-cores are another microarchitecture). `zdi` is the
  interesting one there (expected positive); `sco`, `v1var`, `de-dh` are expected positive too.
- Suggested sizes: `de-dh` 10k (dense enough), `v1var` and `sco` 50k (rare), `zdi` 20k on Intel
  (both arms), Revizor-only on AMD.

## This machine (AMD Ryzen 7 8700F, Zen 4 family 19h model 75h, microcode 0xa705208, SMT off)

Queue `queue_zen4.sh` (started 2026-11-13 23:17, detached; log `queue_zen4.out`):

| # | config | N | arms | expected time | status / result |
|---|---|---|---|---|---|
| 1 | zdi | 20000 | Revizor | ~0.3 h | running |
| 2 | de-dh | 10000 | both | ~2-2.5 h | queued |
| 3 | v1var | 50000 | both | ~5-6 h | queued |
| 4 | sco | 50000 | both | ~8-10 h | queued |

## Notes

- The paper's AMD3 in "Enter, Exit" is Zen 4 family 19h model 74h (Phoenix); this box is model 75h.
- CT-DH on AMD is the strongest contract that held for #PF in Speculation at Fault; for #DE it is the
  next-stricter contract, which is why `de-dh` has violations (a dense, known-answer benchmark).
- `v1var` violations need a check that they are not plain v1 at a deeper nesting (CT-COND model runs at
  the configured nesting) before calling them V1-Var.
- Next (tier B/C, need actor templates): TSA-SQ (kernel->user, U-bit) and TSA-L1 (VM->VM, W-bit) on
  Zen 4, incl. a search for leaks that survive microcode 0208 + VERW (`demo/tsa-sq`, `demo/tsa-l1d`);
  SSE FP inside multi-actor templates (does the FP suppression state cross a privilege / VM switch?).
