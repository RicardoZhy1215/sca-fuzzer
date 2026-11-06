# ctxrl synthetic test (planted world, no hardware), 60k programs each

World: partial patterns q = #(store -> same-address-register load within 3); full pattern = a
rare instruction in between, violates w.p. 0.5 per run. Stages from prior-program percentiles:
spec-filtered q < 4, obs-filtered 4 <= q < 7, near miss q >= 7 or full. Reward = new violation
(dedup by program hash).

Round 1 (ctx.log / fresh.log): the context policy did NOT learn mode selection (fresh share
after a near miss rose 0.90 -> 0.95), both ~0.12 new V/program. Causes: (1) self-imitation also
imitated the MODE token of the positives, which mostly came from the majority mode (fresh) ->
entrenched it; (2) world bug: near miss was nearly every program (q >= 2).
Fixes: SIL excludes MODE tokens (mode learned by advantage only); percentile stage thresholds.

Round 2 (ctx2.log / fresh2.log), last 20k-program window at 56k programs:
| previous outcome | learned mode mix fresh/rerun/edit | new-V rate fresh / rerun / edit |
|---|---|---|
| spec-filtered | 0.99 / 0.00 / 0.01 | 0.112 / 0.000 / 0.006 |
| obs-filtered | 0.97 / 0.01 / 0.02 | 0.127 / 0.000 / 0.014 |
| near miss | 0.63 / 0.03 / 0.34 | 0.115 / 0.231 / 0.298 |
| new violation | 0.28 / 0.04 / 0.69 | 0.109 / 0.000 / 0.443 |
new V per program: context 0.173-0.179 vs fresh-only 0.116 (x1.5); mode selection kicked in
after ~40k programs (~3-4 h of hardware time at ~180 programs/min).
