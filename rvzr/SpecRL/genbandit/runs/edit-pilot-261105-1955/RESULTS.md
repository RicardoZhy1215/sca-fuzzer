# Edit pilot: "write, then revise from the hardware result" -- is the feedback worth acting on?
2 h (2026-11-05 19:55-21:55), coupled-incumbent v4 config, tcenv --policy edit (EditProbeActor,
random edits, no learning). 21539 tc, 7423 episodes, 581 violations, 290.5 V/h. analysis.txt.

Violations per 1000 test cases (95% CI):
| base program outcome | rerun (same program, new inputs) | redraw 1 instruction | change 1 register |
|---|---|---|---|
| fresh programs (7423) | 20.7 (17.5-24.0) | | |
| spec-filtered (3077) | 10.8 | 6.9 | 9.3 |
| obs-filtered (3682) | 18.4 | 17.4 | 16.7 |
| near miss: passed obs filter, no V (510) | 31.5 (19-44) | 40.3 (29-52) | 53.3 (40-66) |
| violated (154) | 262 | 210 | 224 |

Near-miss bases by the base's excess ops (dispatched - retired, max over inputs):
<60: 32.0/1000 (297 bases), 60-120: 45.6 (150), >=120: 89.9 (63) = x4.3 fresh (Fisher p 1.5e-11);
>=120 vs <60: x2.8, p 3.5e-6. Within >=120, edits 84 vs rerun 109 (n.s.).
Fisher: near-miss edit_op vs fresh x2.57 p 3.6e-9; vs near-miss rerun x1.69 p 0.03 (one of
several comparisons); any edit vs rerun x1.49 p 0.077; near-miss rerun vs fresh x1.52 p 0.06.
mem vs reg operand edits: equal (53.4 vs 53.2).
Fresh programs by their own excess: <=0 -> 0/3077; (0,60] 28.5; (60,120] 55.3; >120 46.8.

Conclusions
1. The hardware observation is worth acting on: outcome + excess ops pick out programs whose
   follow-ups violate 4x more often than fresh programs; filtered programs should be dropped.
2. Most of that is SELECTION (which program to keep spending test cases on); random edits add
   at most ~1.5-1.7x over re-running the same program, and nothing for the best programs.
3. Single edits of a violating program keep the violation 21-22% of the time (rerun 26%):
   variants are cheap.
4. V/h unchanged (saturated on v4); the per-program gain needs a non-saturated target to show
   in V/h.
