# GRU vs Transformer, repeated (v4 default64, rule2, from scratch, same-time A/B, seed 11)

| new violations in 10,000 programs per arm | GRU | Transformer | TF / GRU | p |
|---|---|---|---|---|
| A/B 1 (paper, ctx-ab-gru-tf-default64-261107) | 174 | 159 | 0.91 | 0.35 |
| A/B 2 (this run, seed 11) | 177 | 177 | 1.00 | 1.00 |
| separate runs (time-scratch-v4-261112-1928, not same-time) | 138 | 179 | 1.30 | - |
| mean +- sd over the three | 163 +- 22 | 172 +- 11 | | |
| same-time A/Bs pooled | 351 | 336 | 0.96 | 0.59 |
| all three pooled | 489 | 515 | 1.05 | 0.43 |

This run: whole 17.7 vs 17.7 per 1000; second half 111 vs 103 (TF/GRU 0.93, p = 0.63). KL at the
end: GRU 0.080, TF 0.096. Update time: GRU 1.47 s, TF 3.42 s (2 learner cores each).
Against Revizor (111 per 10,000): GRU 1.24-1.59x, TF 1.43-1.61x.

Reading: no difference between the two policy networks. The paired differences (TF - GRU: -15, 0, +41)
change sign, and their spread (sd 29) is larger than their mean (+9). Keep the GRU: it has 2.3x fewer
parameters and its updates are 2.3x faster.
