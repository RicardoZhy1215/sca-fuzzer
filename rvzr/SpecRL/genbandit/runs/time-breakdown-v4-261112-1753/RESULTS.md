# Spectre v4 time breakdown: Revizor vs RL (GRU, warm-started)

Config v4/c441d47e7c.yaml (default64: v4.yaml knobs, 64 instructions, 1 BB, SSBD off), -i 50,
3000 programs each, fuzzer on cores 1-3. Revizor: unmodified `revizor fuzz` driven in-process
(`time_breakdown.py revizor`). RL: tcenv --policy ctx --ctx-policy learn --ctx-mode rule2,
warm-started from ctx-ab-gru-tf-default64-261107/gru/policy.pt (late-training behaviour), learner
on cores 4-7. Both are timed by the same hooks (exclusive time per phase x component, attached at
run time; no Revizor source changed). Measured hook cost: 0.84-0.88 us per call, 0.01% of wall.
Figure: figs/v4-time-breakdown.{png,pdf,csv,txt}.

| ms per program | Revizor | RL (GRU) |
|---|---|---|
| program generation | 9.7 | 10.9 |
| policy (RL: get + install the agent's program, send outcome) | - | 0.7 |
| input generation | 1.4 | 1.4 |
| filters: hardware | 62.7 | 65.9 |
| filters: other (fenced copy assembly, compare) | 19.1 | 20.7 |
| fast path: model (contract trace + taint + boosting) | 20.7 | 41.1 |
| fast path: hardware | 29.1 | 55.6 |
| confirmation (nesting, taint, priming, noise, arch) | 323.5 (69.3%) | 963.1 (82.7%) |
| RL harness (tcenv features / hw-observation summary) | - | 3.4 |
| other | 0.3 | 0.9 |
| **total** | **466.9** | **1164.4** |

- Violations: Revizor 26 (8.7 per 1000 programs, 66.8 V/h), RL 76 (25.3 per 1000, 78.3 V/h).
- Confirmation costs the same per violation on both: 37.3 s (Revizor) vs 38.0 s (RL). RL spends
  2.5x longer per program because 2.9x more of its programs violate.
- The fast path costs RL twice as much per program (model 41 vs 21 ms, hardware 56 vs 29 ms),
  because more of RL's programs pass both filters.
- The RL machinery on the fuzzer's critical path is 0.7 ms (policy) + 3.4 ms (tcenv
  features/logging) = 4.1 ms per program, 0.35% of RL's wall time. PPO updates run in the agent
  process on cores 4-7 and are off the critical path. Existing A/B logs (ctx-ab-gru-tf-default64-
  261107): 1.38 s per GRU update (3.26 s Transformer), one per 256 programs.

Compared with earlier claims:
- "Confirmation ~79% of time" is the RL figure (A/B logs 77-79%; here 83% warm). For Revizor at
  its lower violation rate it is 69%.
- "V/h saturated by confirmation": RL gets 1.17x Revizor per hour here (78.3 vs 66.8), compared
  with 2.9x per program. Close to the earlier "about equal per hour", but not identical.
