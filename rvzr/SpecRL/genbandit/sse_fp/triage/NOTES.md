# SSE-FP normal arm, 20000 programs each: are the violations real, and are RL's one gadget?

Sample: all 8 Revizor violations + 20 of RL's 187 (random, seed 0, `rl-sample-20.txt`).

## 1. Real? (`sample-normal-20k.tsv`, `../triage.py`)
Hardware replay on each violation's own saved pair, with variants. The port reproduces the old
tool's verdict on a known direct-VVI violation (side-channel-fuzzer spec-ext 201843: FPVI-family,
A-only set 46).

| verdict | Revizor | RL |
|---|---|---|
| FPVI-family (gone with FTZ+DAZ) | 7/8 | 17/20 |
| ftz-architectural (FTZ/DAZ changes architectural results; inconclusive) | 1/8 | 3/20 |
| architectural (survives lfence after every instruction) | 0 | 0 |
| not reproduced | 0 | 0 |

All 28 reproduce, and all 28 disappear with an lfence after every FP op. None is a model/hardware
architectural mismatch.

## 2. One gadget? (`minimized-normal-20k.tsv`, `../minimize.py`)
Each candidate program during minimization gets its own pair (A, and B with the bytes the candidate's
taint marks copied from A), because the stored pair is equivalent only for the original program.
All 28 minimize (5-17 instructions; one RL gadget keeps 44).

- Distinct shapes (registers renamed, numbers dropped): Revizor 8/8, RL 20/20. RL's violations
  are not copies of one gadget.
- They form a narrow family, though. FP ops in RL's gadgets: mulpd 16/20, divpd 8/20. Revizor's
  spread over divsd 3, divss 2, mulpd 2, mulsd 2, divpd 2. FP->GPR transmitter: RL extractps 11/20,
  movq 8/20; Revizor movq 4/8, pextrw 3/8, never extractps.
- Structure, by hand. The labels below are structural only; the mechanism (where the secret is)
  needs the input-diff pass, which has not been run.
  - "two-level shape" (FP value -> address -> load -> loaded value -> address):
    Revizor 3/8 (212835, 213435, 214334); RL 9/19 shown (222321, 225432, 232608, 232639,
    232717, 232801, 234304, 235333, 001830).
  - "one-level shape" (FP value -> address directly; direct VVI, or gated if an earlier dead FP op
    carries the secret): Revizor 5/8, of which 3 have a dead FP op before the probe (212545, 215209,
    215258); RL 9/19, of which 5 have one (223736, 224459, 230021, 233740, 001651).
  - RL 003022 is unclear (the address mixes the FP value and a loaded byte); 230806 (44
    instructions) was not read.
- 232608 and 232639 have the same core (mulpd -> movq -> cmovnp load -> store) and differ only in
  leftover instructions in front: the closest pair in the sample.

Reading: RL found 22x as many violations per program, and they are not one gadget repeated.
But they come from a narrower family than Revizor's (mulpd/divpd probe, extractps/movq
transmitter), with about the same mix of one- and two-level structure.

## 3. The suppression (205811 / gated) shape? (`gates-normal-20k.tsv`, `../gate_scan.py`)
Port of side-channel-fuzzer's gate_scan.py, on min.asm with the pair rebuilt against it. GATE = an
FP op whose operation is needed (replacing it with a plain move of the same operands removes the
difference), whose result is dead, and whose effect survives 512 nops after it.

| | gated | gate ops |
|---|---|---|
| Revizor | 4/8 (211533, 212545, 215209, 215258) | divss, mulss, divpd, mulsd |
| RL | 7/20 (223736, 224459, 230021, 233740, 001651, 002414, 003022) | divps x2, mulpd x3, divpd x2 |

Every gated one has the presence/absence shape in its original hardware traces: the extra cache
set appears in one input only, as in 205811. The earlier 21-violation study listed gates from
mulss/mulpd/mulps/divps/divpd (plus 205811's denormal input). divss and mulsd as gates (both in
Revizor's sample here) were not in that list.
Extrapolated, RL produced about 0.35 x 178 = 60 gated violations in 20000 programs and Revizor
about 4. Still to check: whether the secret is in the gate's operands (input-diff pass).
