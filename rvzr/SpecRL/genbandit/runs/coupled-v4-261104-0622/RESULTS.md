# Coupled (per-test-case, interaction-aware) knob policy, v4 -- 12 h (2026-11-04 06:22-18:23)

`tcenv.py --mode bypass --policy coupled --minutes 720` (alpha 0.15, cand_frac 0.5), learner =
`coupled.py learn` on cores 4-7 (694 rounds, 2.35 s per fit). Summary: analysis.txt, insight.txt.

## Headline
| | V/h | notes |
|---|---|---|
| v4.yaml control (v4-rl run) | 25.1 | |
| v4 bandit, 12 h average / last 2 h | 90.8 / 160 | runs/v4-rl-261103-0208 |
| frozen bandit argmax (old incumbent), live here before promotion | 196.3 (180-212) | n=75140, V=587, 7.8 V/1000 tc, 419 tc/min |
| new incumbent, live after promotion | **283.1 (271-295)** | n=108006, V=2281, 21.1 V/1000 tc, 223 tc/min |
| new incumbent as candidate, interleaved with the old one before promotion | 277.8 (182-374) | n=1487, V=32 -- same window as the 196 |
| whole run | 256.6 | 203721 tc, 3080 V (3.7 h at the old incumbent) |
Promotion at round 214 (3.7 h): lcb 203.5 > incumbent mean 196.2. Exploration cost: overall /
incumbent-track V/h = 0.995. Policy overhead 0.48 ms/tc.

Old -> new: n_regs 4->6, entropy 10->16, binary on->off, spec_filter off->on.
Model attribution at the new incumbent (one knob reverted; each within ~noise on its own):
spec_filter off x0.95 (P(V) same, non-V tc 49->62 ms: the filter only saves the fenced run),
binary on x0.87, entropy 10 x0.88, n_regs 4 x0.91 -> product ~1.5, matches the live 1.44.

## Interactions
None clears the noise floor (top: program_size x locks 0.24, members 10-90% 0.12-0.62; prior-only
floor median 0.11, p90 0.17). Best with interactions == best with main effects only. Near the
optimum the knobs act as independent multiplicative factors (e.g. locks on = x2.3 at every
program size). The gain over the bandit comes from per-test-case switching / interleaved
comparison resolving 5-15% effects that 72 ten-minute slices could not.

## Other findings
- n_inputs: 50 inputs give 4x P(V) per test case (89.7 vs 21/1000) but confirmation time scales
  with inputs (T_v 10.4 / 20.8 / 40.4 s for 10/25/50), so V/h drops to x0.28.
- entropy has an interior optimum at 16 (4: x0.74, 10: x0.88, 24: x0.90).
- Saturation: the new incumbent spends 81.8% of its time confirming violations, so better
  configurations can add at most x1.22 more; the remaining lever is the ~10 s confirmation cost.

## Not yet done
- Dedicated 50/50 interleaved A/B old vs new incumbent (the pre-promotion interleaved estimate
  has a wide CI; the tight 283 is from a later window).
- SSBD A/B on sampled violations of the new incumbent (the v4 run's 21/21 was for other configs).
