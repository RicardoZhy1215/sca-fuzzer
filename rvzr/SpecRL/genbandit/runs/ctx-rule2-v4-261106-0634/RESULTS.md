# ctxrl rule2 (fixed modes) + lineage reward + warm start, v4, 2 h (2026-11-06 06:34-08:34)
Modes: violation / filtered -> fresh; near miss -> rerun or edit 50/50. Reward: first violation
per lineage. Generation policy warm-started from ctx-learn-v4-261105-2314 (7.3 h of training).
15042 programs, 607 new violations (= lineage-new = hash-new = all violations; 0 duplicates),
303.7 new V/h, 125 programs/min, overhead 0.58 ms/tc. analysis.txt.

New V per 1000 programs: 40.4 overall; by 30-min block 39.3, 40.2, 40.3, 41.8.
  fresh 37.2 (13364) | rerun after near miss 75.4 (875) | edit after near miss 54.8 (803)
Comparison (lineage-dedup, per 1000 programs; different runs/days, not interleaved):
  Revizor fresh 21-22 | rule baseline (Revizor generation + old rule) 22.3 | progrl last 4 h 33.4
  | ctx learned modes 30.6 | this run 40.4 (x1.85 Revizor).
V/h 303.7 vs ~290 before: v4 stays near its confirmation-time ceiling.
