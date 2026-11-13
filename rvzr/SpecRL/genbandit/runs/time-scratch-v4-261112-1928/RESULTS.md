# Case study: time breakdown of ctxrl trained from scratch on Spectre v4, GRU vs Transformer

Config v4/c441d47e7c.yaml (default64), -i 50. ctxrl (tcenv --policy ctx --ctx-policy learn
--ctx-mode rule2), NO warm start, 10000 programs each. Runs were sequential, not a same-time A/B:
GRU 19:28-21:31, then Transformer 21:31-23:55. Fuzzer on cores 1-3, learner on 4-7. Transformer:
2 layers, 4 heads, FFN 512 (as in ctx-ab-gru-tf-default64-261107). Timing hooks:
time_breakdown.py (exclusive time per phase x component, no Revizor source changed; hook cost
0.01% of wall). Revizor reuses time-breakdown-v4-261112-1753/revizor (3000 programs, same config);
its per-program cost is stationary, since Revizor does not learn.
Files: report.txt, blocks.txt, latency.txt; figures figs/v4-time-breakdown-scratch.{png,pdf,csv,txt}
and figs/v4-time-blocks.{png,pdf,csv,txt}.

## (a) Breakdown per program over the whole run (ms/program, share of wall)
| | Revizor (3000) | RL GRU (10000) | RL Transformer (10000) |
|---|---|---|---|
| program generation | 9.7 | 11.1 | 11.1 |
| policy: waiting for the agent | - | 0.15 | 0.62 |
| policy: install / report outcome | - | 0.15 | 0.16 |
| input generation | 1.4 | 1.4 | 1.5 |
| filters (hardware + fenced copy) | 81.8 | 87.6 | 85.4 |
| fast path (model + hardware) | 50.2 | 79.0 | 88.8 |
| confirmation | 323.5 (69.3%) | 551.6 (75.0%) | 670.7 (77.8%) |
| RL harness (features, logging) | - | 3.3 | 3.4 |
| other | 0.3 | 0.8 | 0.9 |
| **total** | **466.9** | **735.1** | **862.6** |
| violations | 26 (8.7/1000) | 138 (13.8/1000) | 179 (17.9/1000) |
| per hour | 66.8 | 67.6 | 74.7 |
| confirmation per violation | 37.3 s | 40.0 s | 37.5 s |

## (b) Over training (per 1000 programs; blocks.txt, figs/v4-time-blocks)
Generation (~13 ms), filters (~85 ms) and RL overhead (~4 ms) stay flat. What moves the total
is confirmation, which follows the violation rate: per block, confirmation share 68-80% (GRU) and
73-84% (TF), and violations 10-20 (GRU) and 12-28 (TF) per 1000. The fast path grows somewhat
(GRU 65 -> 83 ms, TF 62 -> 95 ms), as more programs pass the filters.

## (c) Learner and policy on the critical path
| | GRU | Transformer |
|---|---|---|
| parameters | 416,277 | 958,357 |
| PPO updates (one per 256 programs) | 39 | 39 |
| time per update | 1.10 s (max 1.15) | 2.27 s (max 2.37) |
| total learner time | 43 s = 0.6% of wall | 88 s = 1.0% of wall |
| sampling one program (batch 1, ~160 decisions, cores 4-7, final policy) | 37 ms median | 68 ms median |
| fuzzer blocked waiting for the agent | 1 time, 1.5 s (agent start-up) | 17 times, 6.2 s total |
| policy + RL harness on the critical path | 3.6 ms/program (0.5%) | 4.1 ms/program (0.5%) |
Updates run in the agent process on other cores, so they are off the fuzzer's critical path.
Sampling (37-68 ms) is also off it: with 16 interleaved episodes the agent writes ahead while
the hardware runs. The fuzzer waited only at start-up (GRU), and 16 more times for TF (6.2 s in
total, 0.07% of wall), when a slower sample met a run of fast, filtered programs.

## Learning (vs the earlier same-time A/B, ctx-ab-gru-tf-default64-261107)
| new violations | first 5000 | second 5000 | total 10000 |
|---|---|---|---|
| GRU, this run | 63 | 75 | 138 |
| TF, this run | 85 | 94 | 179 |
| GRU, A/B | 66 | 108 | 174 |
| TF, A/B | 69 | 90 | 159 |
KL to the prior is similar in all four runs (~0.1 by update 20; final 0.104 GRU / 0.095 TF here).
The order of GRU and TF flips between the two experiments. A single 10k run does not separate the
architectures, and the GRU's gain over Revizor varies run to run (second half 75 here vs 108 in the
A/B; Revizor about 54 per 5000). Repeated runs are needed for a claim about the size of the gain.

## Estimate for the earlier GRU / Transformer A/B run (ctx-ab-gru-tf-default64-261107, first 10000 programs per arm)
That run logged each program's outcome stage and measured time (tc.jsonl `s`), but not the per-stage
split. Estimate = each program's MEASURED time, split by the share of each group for that outcome
(spec_filter / observ_filter / fast_path / violation / fp_*) in this run's programs.jsonl. So totals
equal the measured ones. Check: predicting totals from this run's per-outcome means is +6.6% (GRU)
and +1.0% (TF) off; the old run was 3-6% faster per outcome (e.g. a violation 37.7 s vs 38.8 s).
| ms/program | old GRU | old Transformer |
|---|---|---|
| Program generation | 10.4 (1.3%) | 10.6 (1.4%) |
| Policy (RL), incl. waiting | 0.1 | 0.6 (0.1%) |
| Input generation | 1.3 | 1.4 |
| Filters | 82.6 (9.9%) | 81.9 (10.6%) |
| Fast path | 83.2 (10.0%) | 81.1 (10.5%) |
| Confirmation | 650.7 (78.2%) | 595.8 (76.8%) |
| RL harness + other | 4.0 (0.5%) | 4.1 (0.5%) |
| total (measured) | 832.3 | 775.4 |
Directly measured in the old run: tcenv's RL overhead per program (ovh_ms, incl. waiting for the agent)
0.76 ms on average (0.09% of time), max 1.5 s (agent start-up), even with only 2 learner cores per arm.
Numbers: estimate-old-ab.json.
