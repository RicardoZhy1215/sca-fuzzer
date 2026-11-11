# SSE-FP normal arm (entropy 31), Zen 4: every violation analysed

Runs: `runs/revizor-ssefp-normal-20k-261110-2112` (unmodified Revizor, 20000 programs, 8 violations),
`runs/ctx-ssefp-normal-20k-261110-2112` (ctxrl GRU, learned modes, xmm families, 20000 programs,
187 violations: 178 new + 9 reruns of a violating program). Same config body
(`configs/ssefp-45c051fb1c.yaml`). Pipeline (`analyze_all.sh`): triage.py -> minimize.py -> classify.py.

## 1. Real?
| verdict (hardware replay on the saved pair) | Revizor | RL |
|---|---|---|
| FPVI-family (gone with FTZ+DAZ) | 7 | 176 |
| ftz-architectural (FTZ/DAZ changes architectural results; inconclusive) | 1 | 9 |
| architectural (survives lfence after every instruction) | 0 | 0 |
| not reproduced on the saved pair | 0 | 2 |
No violation is a model/hardware architectural mismatch.

## 2. Mechanism (classify.py: roles by gate_scan, secret by delta debugging on the hardware, flow by model + static taint)
| class | Revizor | RL | RL / Revizor per program |
|---|---|---|---|
| gated (205811 suppression: the secret enters a non-probe FP op whose underflow suppresses a later VVI probe) | 4 | 93 | 23x |
| two-level FPVI (the secret is a word only a transient load reads) | 3 | 84 | 28x |
| direct VVI (the secret is the probe's own operand) | 1 | 8 | 8x |
| unresolved (flaky: original not reproducible / secret search did not converge) | 0 | 2 | |
For comparison, side-channel-fuzzer's spec-ext normal arm (10 violations, 3 h) was 6 gated / 3 two-level / 1 direct.
8 RL violations were classified by hand as gated (all 8 keep the difference with 128 and 512 nops after the gate, 3/3 each: the long-lived state, not a window effect) (230949, 231028, 231357, 232030, 232155, 232938,
233107, 235805). In each, the secret enters a non-probe FP op whose result is read later but never
reaches an address. gate_scan's "dead" test misses these. 232938 is a two-op chain: divpd xmm4 feeds
divpd xmm3, xmm4, and that result is dead.

Over RL's training: programs 0-8000: 4 violations, all two-level. 8000-14000: 29 gated, 23 two-level.
14000-20000: 64 gated, 57 two-level, 8 direct. First gated at program 8141, first direct at 14684.
Revizor's first gated came at program 1443, its first direct at 19547.

## 3. What the gadgets are made of
- Gate ops. RL: mulpd 57, divpd 26, mulsd 6, divps 3, divsd 1. Revizor: divss, mulss, divpd, mulsd.
  The old 21-violation study had gates from mulss / mulpd / mulps / divps / divpd (plus 205811's
  denormal input). mulsd (RL 6, Revizor 1), divsd (RL 1) and divss (Revizor 1) gates are new
  instances. Gated secrets in RL: 65 in a register's low 64-bit lane, 37 in the high lane: the
  upper lane of a packed gate (mulpd/divpd) sets the state as well.
- Two-level FP->GPR transmitter. RL: extractps 30, movq 28, pextrw 17, cvtss2si 4, pmovmskb 4,
  cvtsd2si 1. Revizor: movq 2, pextrw 1; Revizor never used extractps here.
  In 224014 (read by hand) the second level goes through EFLAGS: the transient load's value sets
  the flags of an sbb, and a cmovnl picks the next address. 47 of the 84 have some flags consumer
  after the transmitter (upper bound; not verified one by one).
- Diversity: all 20 sampled RL gadgets have distinct shapes. RL gated: 18 distinct (gate op,
  mul/div probe) pairs; two-level: 10.

## Reading
RL produced every class Revizor did, at 8-28x the rate per program. Of the late-training
violations about half are gated (the suppression state), the class most specific to this
platform. No mechanism outside the FPVI family / suppression state appeared. The new things are
instances: new gate ops (mulsd, divsd, divss), live-but-untransmitted gates, chained gates, and
flag-based second levels.

## Counting: 178, not 187
The paper counts a violation once per lineage, like the RL-performance figure (steps.jsonl
new_v = 178). Revizor's log has 187 violations: the other 9 are reruns (6) or edits (3) of a
program that had already violated: 223736, 231028, 232435, 232639 (a rerun of 232608), 235717,
000344, 000425, 002614, 003101. Over the 178 new ones: gated 89, two-level 81, direct 6,
unresolved 2. Gate ops: mulpd 54, divpd 26, mulsd 6, divps 2, divsd 1. Secret lanes: lo 63,
hi 35. Live-but-untransmitted gates: 7 (231028 is a rerun).
