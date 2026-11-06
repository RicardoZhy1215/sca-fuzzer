# v4 RL run (2026-11-03 02:08 – 14:10, full 12 h)

Setup: `rl.py --space v4 --hours 12 --slice-min 10 --lr 0.1 --eps 0.1`, action space `space_v4.py`
(SpecRL hierarchical_testing/v4.yaml fixed part: CT-SEQ, 1 BB, no jcc, SSBD off, Unicorn, base.json,
random seeds; 11 knobs). Slice 0 and every 6th slice = control (v4.yaml knobs, `-i 50`).
Slice length: median 10.1 min, max 11.5 min (the timeout is only checked between test cases).

Control choice: 20-min baselines in `../v4-baseline-261103`: `-i 50` 20.7 V/h, `-i 10` 14.8 V/h
(3.2x test cases, but the speculation filter drops 91% instead of 78% and priming rejects 9), so
v4.yaml as is (`-i 50`) is the stronger control.

| track   | slices | hours | violations | V/h  | tc/min | V per 1000 tc |
|---------|--------|-------|------------|------|--------|---------------|
| policy  | 59     | 10.00 | 908        | 90.8 | 781    | 1.94          |
| control | 12     | 2.03  | 51         | 25.1 | 722    | 0.58          |

Control slices: 12–45 V/h (mean 25.1). Policy V/h per block of 12 slices (in slice order):
37 → 63 → 66 → 97 → 129 → 160. Best slice 61: 258.5 V/h (64 instr, mem 0.75, 6 regs, entropy 10,
`-i 10`, spec filter off).

Final policy: program_size 64 = 0.81, mem_ratio 0.75 = 0.73, n_inputs 10 = 0.74, value_mix
default = 0.93, semaphore off = 0.82, locks on = 0.79, spec_filter off = 0.62, entropy 10 = 0.57,
n_regs 4 = 0.54, binary on = 0.64, flags 0.50/0.50.

Per-knob marginals over policy slices (V/h): program_size 8: 0, 20: 28, 40: 64, 64: 130 ·
mem_ratio .25: 9, .5: 85, .75: 110 · entropy 4: 99, 10: 97, 16: 76, 24: 49 · n_regs 2: 62,
4: 120, 6: 82 · value_mix default: 111, full64: 46 · semaphore off: 103, on: 70 · spec_filter
off: 98 (2.81 V/ktc), on: 76 (1.09 V/ktc). Marginals are biased toward values the policy picked
late, when the other knobs were already good.

## Validation: are they store bypass?

`ssbd-ab/`: 21 random violations (14 policy, 7 control), each reproduced twice with its
minimize.yaml (`-n` = the slice's n_inputs) with SSBD off and twice with SSBD on
(`x86_executor_enable_ssbp_patch: true`). All 21: 2/2 with SSBD off, 0/2 with SSBD on, no errors.

## Conclusions

1. Unlike v1, the gain is program quality, not throughput: test cases/min are about the same,
   violations per test case are 3.3x the control.
2. The policy learned the store-bypass mechanism: long programs, many memory accesses, low address
   entropy (more store->load pairs to the same address).
3. `full64` inputs ignore `data_generator_entropy_bits` (full 64-bit random words), so value_mix
   and entropy are not independent knobs. This is a flaw in the action space.
4. The best slices log one violation every ~14 s (median), close to the confirm+save cost that
   capped v1 at ~350-400 V/h, so V/h will saturate here too.
5. Not yet compared against a hand-tuned Revizor config, and the policy numbers include
   exploration. Next: fix the argmax config for ~1 h vs the control and a hand-tuned config.
