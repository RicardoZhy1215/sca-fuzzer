# ctxrl synthetic, GRU vs Transformer policy (2026-11-06, no hardware)

`ctxrl.py synthetic --fresh-only --programs 40000 --arch {gru,tf}`, seed 0, same hyperparameters
(lr 3e-4, KL 0.1, SIL 1.0), 2 cores each (gru 4-5, tf 6-7). Planted world of ctx-synth-261105:
violation = store -> same-address-family load within 3 instructions with a rare instruction in
between, rewarded w.p. 0.5. Transformer: 2 layers, width 128, 4 heads, FFN 512 (718k params vs
266k). Rate = new violations per program over the last 4000 programs (SE ~0.004).

| programs | GRU | Transformer |
|---|---|---|
| 5120 | 0.015 | 0.015 |
| 10240 | 0.061 | 0.050 |
| 20480 | 0.084 | 0.067 |
| 30720 | 0.093 | 0.072 |
| 38400 | 0.107 | 0.075 |

The GRU curve reproduces ctx-synth-261105/fresh2.log exactly (0.0843 at 20480). The
Transformer learns more slowly and plateaus ~25-30% lower here. One seed each; the planted rule is
local (within 3 instructions), so it does not need long-range attention, and the Transformer
reuses hyperparameters tuned for the GRU. Wall time per program ~2x (19.6 vs 9.5 min).
