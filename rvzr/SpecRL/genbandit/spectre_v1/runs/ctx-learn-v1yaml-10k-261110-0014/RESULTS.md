# ctxrl (GRU, learned modes) vs unmodified Revizor on configs/v1.yaml, 10000 programs

Setup: `tcenv.py --space v1 --action '{"program_size": 32, "n_bb": 2}' --policy ctx --ctx-policy learn
--ctx-mode learn --max-programs 10000`. The config is byte-identical to configs/v1.yaml (CT-SEQ,
SSBD on, 2 basic blocks, 32 instructions, -i 50). Revizor's 10000 programs come from three joined
runs (1824 + 1175 + 7000). Figure: `figs/revizor-vs-ctx-learn-v1yaml-10k.{png,pdf,csv}`.

| | programs | violations | per 1000 |
|---|---|---|---|
| Revizor | 9999 | 729 | 72.9 |
| RL, new violations (first per lineage) | 10000 | 748 | 74.8 |
| RL, every violation (incl. 62 reruns/edits of an already violating program) | 10000 | 810 | 81.0 |

The figure counts RL's new violations: a rerun or edit of a program that has already violated
finds the same bug again.

- Whole run: RL / Revizor = 1.03x, conditional binomial p = 0.64, i.e. no difference.
- Second half: 1.13x, p = 0.11, not significant, and confounded by time. Revizor's part 3
  (programs 3000-10000) ran after the RL run, and Revizor's own rate fell from 81 to 70 per 1000
  between part 1/2 and part 3, so machine drift alone could explain the gap.
- Per 2500 programs, Revizor / RL: 79.6 / 74.4, 75.2 / 70.8, 67.6 / 71.6, 69.2 / 82.4.

Learning: 39 PPO updates. KL to the prior rose to about 0.026 by update 19 and then plateaued
(0.037 at the end). Entropy went from 3.66 to 3.61. The mode shares never moved off the prior:
fresh 92.4% (79.4 new V/1000), rerun 3.5% (11.6), edit 4.2% (24.1). The policy did not learn to
avoid reruns and edits even though they find few new violations.

Why there is no gain: on v1 (CT-SEQ) about 8% of random programs violate, and the program
features we extract barely predict which do (best single-feature AUC 0.54 over 3527 fresh
programs, vs 0.64 on v4). The reward therefore gives the policy no direction. A harder target
(e.g. CT-COND, where plain v1 is allowed by the contract) is needed for program-level RL to matter.
