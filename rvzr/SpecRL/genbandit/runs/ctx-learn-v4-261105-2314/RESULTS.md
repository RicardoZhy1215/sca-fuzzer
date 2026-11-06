# ctxrl, learned modes (PPO mode head), v4 -- stopped at 7.3 h (planned 12 h)
`tcenv --policy ctx --ctx-policy learn` (modes from the policy head), reward = new violation by
program hash. Stopped early (2026-11-06 06:34, user-approved direction change); final policy.pt
saved and used to warm-start the next run. analysis.txt = analyze_ctx.py.

68787 programs, 2124 hash-new violations (30.9/1000), 290 new V/h (v4 saturated), 157 programs/min,
overhead 0.59 ms/tc. Lineage-dedup (first violation per lineage): 2105 = 30.6/1000.
New V/1000 by 1.2-h block: 26.6, 30.8, 34.1, 33.3, 30.5, 31.0.
Fresh-mode programs 31.1/1000 vs Revizor 21-22 -> the gain is generation quality (like progrl).

Mode selection was NOT learned: ~0.90/0.05/0.05 throughout, not conditioned on the previous
outcome; later it drifted toward "fresh everywhere" (context-free average), although after a
near miss rerun/edit paid ~55-73/1000 vs fresh ~37-40 (lineage-dedup) and after spec/obs-
filtered fresh was best. The rule baseline's apparent 29.3/1000 was 22.3 by lineage (its edits
after a violation were all the same violation). Decision (with the user): fixed mode rule
(violation/filtered -> fresh, near miss -> rerun/edit), lineage reward, RL for generation only.
