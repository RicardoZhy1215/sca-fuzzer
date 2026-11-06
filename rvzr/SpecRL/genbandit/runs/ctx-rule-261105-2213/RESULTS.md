# ctx rule baseline (edit-pilot rule, no learning), 1 h, v4 coupled-incumbent config
Rule: near miss with ex_max >= 60 -> RERUN; new violation -> EDIT (one random edit); else FRESH.
Reward = new violation (dedup by program). E=16 interleaved episodes x N=8.
10110 programs, 296 new violations (0 duplicates), 295.8 new V/h, 168 programs/min,
overhead median 0.61 ms / p99 1.0 ms per test case.
| mode (when) | n | new V/1000 (95% CI) |
|---|---|---|
| fresh | 9533 | 21.9 (19.0-24.9) |
| rerun (after near miss, ex>=60) | 325 | 61.5 (35-88) |
| edit (after a new violation) | 252 | 265.9 (211-320) |
Overall 29.3 new V/1000 = x1.34 over fresh; V/h ~ unchanged (saturated).
