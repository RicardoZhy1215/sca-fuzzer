# Spectre v4 / v1 violations: which known speculation explains them (classify_contract.py)

Sets: the violations behind the paper's v4 and v1 figures (new violations only, RL cut at 10,000
programs; violation_similarity.sets()). Per violation, on its own saved pair, replaying the whole
saved input sequence (the pair's neighbours train the predictors; the pair alone loses most v4
leaks): model = CT-BPAS (v4) / CT-COND (v1) contract traces differ (nesting 1, then 5);
hardware = the leak disappears with SSBD on (v4) / an lfence at every basic-block start (v1).
v1 results: ../../spectre_v1/analysis/contract/classes.tsv.

## v4 (446)
| | explained (store bypass) | not reproduced |
|---|---|---|
| Revizor (113) | 111 | 2 |
| RL GRU (174) | 173 | 1 |
| RL Transformer (159) | 158 | 1 |
Every reproducible v4 violation is store bypass: CT-BPAS explains it AND SSBD removes it. Nothing
model-only, hw-only or unexplained. Not reproduced in 2 + 5 replays: 104716, 120351 (Revizor), 155900 (GRU),
134807 (TF). 144010 (GRU) failed the first 2 replays, reproduces 4 of 7 later, and is explained (counted above).

## v1 (1477)
| | explained (branch misprediction) | other |
|---|---|---|
| Revizor (729) | 726 | 1 hw-only (division), 1 not reproduced, 1 artifact lost |
| RL GRU (748) | 748 | - |
- 204859 and 210030 (Revizor) failed the first 2 replays and then reproduced 7 of 7, explained (counted above).
- 115837: not reproduced in 7 replays.
- 214342: every file of the artifact is empty. It was written at 21:43:42, a minute before the machine's
  unclean restart, so the data never reached the disk.
- 122246, hw-only: block-start lfences remove it, but CT-COND (nesting 1 and 5) does not explain it.
  The speculative path after `jnp` contains `div dword ptr [r14 + rdi]`. Replacing the div with a nop, or an
  lfence right before or right after it, removes the leak (0/5 each; the original is 5/5). So the leak
  needs the division executing on the mispredicted path. This is consistent with the known leak through
  value-dependent division latency (Oleksenko et al., S&P'23), which CT-COND does not model. Not checked
  further (e.g. against Revizor's vspec-ops-div contracts).

## Reading
v4: 442 of 446 are store bypass and none is anything else. v1: 1474 of 1477 are branch misprediction;
one also needs a speculative division. Neither tool found a violation outside the expected speculation
for its target: on v4 / v1 the RL policy finds more of the same leak, not new kinds of leak.
