# SSE floating point: Revizor vs ctxrl

Same campaign as side-channel-fuzzer's `sse_fp/spec-ext` (README sections 8-9 there) and
SpecRL's `hierarchical_testing/sse-fp.yaml`. CT-SEQ, one basic block (no Spectre v1), SSBD on
(no Spectre v4), BASE + SSE categories, the blocked FP arithmetic / conversions / SSE4.1 FP ops
allowlisted by name. Speculation filter off. 64 instructions, 24 memory accesses, -i 50. Random seeds.

- `space_ssefp.py`: the config space (`tcenv.py --space ssefp`). The only knob is `entropy_bits`:
  31 = normal-value arm, 16 = denormal arm.
- `configs/`: the yamls it writes. `ssefp-45c051fb1c.yaml` is the normal arm. Revizor runs on it
  directly, and tcenv writes a yaml with the same body.
- xmm registers: ctxrl's register decisions cover the 6 GPR families plus xmm0-xmm7 (one family
  each). The extra families exist only when the instruction set has xmm operands (`n_fam` 14 here,
  6 for v4 / v1, whose tables and checkpoints are unchanged). Prior = Revizor's uniform choice.
- `runs/launch-normal-20k-*.sh`: Revizor 20000 programs, then ctxrl 20000 programs, then the figure
  `figs/revizor-vs-ctx-ssefp-normal-20k`.

Earlier numbers on this campaign come from side-channel-fuzzer, a different fork (e.g.
analyser_stat_threshold 0.05 there vs 0.5 here): spec-ext 10 violations / 21973 programs (normal),
11 / 21899 (denormal). Use the Revizor run here as the baseline, not those.
