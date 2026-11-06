# Step 2: perturbed collection + predictability go/no-go (2026-11-03 22:46 - 11-04 00:47)

Run: `tcenv.py --mode bypass --perturb 0.1 --minutes 120` on the frozen argmax config of
runs/v4-rl-261103-0208 (64 instr, mem 0.75, n_regs 4, entropy 10, -i 10, default mix,
binary on, flags off, semaphore off, locks on, spec filter off).
Analysis: `analyze_tc.py` -> analysis.txt; `step2_extra.py` -> analysis_extra.txt.

## Collection
51264 test cases, 373 violations, 186 V/h (base-only, time-weighted: 191 V/h; step-1 frozen = 188).
Time share: violation confirmation 55.6% (10.7 s each), observ-filtered 36.1% (54 ms each).

## Predictability (5-fold CV; AUC SE <= ~0.027 for violations)
| label | config | config+feats LR | config+feats MLP | feats|base LR | feats|base MLP |
|---|---|---|---|---|---|
| violation (n+=373) | 0.518 | 0.612 | 0.563 | 0.602 | 0.559 |
| passed obs filter (n+=3146) | 0.517 | 0.601 | 0.590 | 0.588 | 0.574 |
Same-config signal is real but weak (AUC 0.60, ~4 SE; AP 0.0113 vs base rate 0.0074).
Top decile lift 2.0x, bottom two 0.5-0.6x. The MLP is worse than logistic (overfits 341 positives).
Strongest single feature: ld_to_addr (AUC 0.58; 17.3 vs 15.8 per program).

## Skip action: NO-GO
Best simulated gain x1.026 (skip 20%), within noise.

## Ceiling
V/h = y / (a + y c), a = 62 ms/tc, c = 10.7 s/violation: per-tc yield x2 -> V/h x1.29;
x3 -> x1.42; infinite -> x1.80. The v4 reward is near saturation, as v1's was.

## One-knob perturbations (~130-480 tc each; mostly too small)
program_size < 64, mem_ratio < 0.75, n_regs 2, entropy 16: 0 violations (consistent with the policy).
binary=off: 10 V / 475 tc = 21/1000 vs base 7.4 (Poisson p ~0.003 alone, ~0.05 after 18 comparisons),
V/h 259 (95% ~98-419) vs 191 -> candidate for a dedicated A/B. entropy 4: 4 V / 167 tc.
