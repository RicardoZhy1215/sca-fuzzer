# v1 RL run (2026-11-02 22:19 – 11-03 00:45, stopped by hand at 2.40 h of 6)

Setup: `rl_v1.py --hours 6 --slice-min 3 --lr 0.1 --eps 0.1` (now `rl.py --space v1`), action space `space_v1.py`
(SpecRL v1.yaml, CT-SEQ, Unicorn, base.json, random seeds). Slice 0 and every 6th slice =
control (v1.yaml knobs, `-i 50`). The in-flight slice 43 was killed and is not in slices.jsonl.

| track   | slices | hours | violations | V/h   | tc/min | V/tc  |
|---------|--------|-------|------------|-------|--------|-------|
| policy  | 35     | 1.86  | 418        | 224.5 | 49.9   | 0.075 |
| control | 8      | 0.53  | 38         | 71.1  | 11.0   | 0.108 |

Policy first third 162.7 V/h, last third 259.5 V/h.

Per-knob marginals over policy slices (V/h, tc/min, V/tc):

| knob         | values |
|--------------|--------|
| n_inputs     | 10: 355, 70.2, .084 · 25: 158, 34.3, .077 · 50: 84, 37.4, .037 |
| program_size | 8: 216, 21.9, .164 · 16: 166, 74.0, .037 · 24: 247, 28.6, .144 · 48: 273, 101.2, .045 |
| mem_ratio    | .25: 183 · .5: 252 · .75: 197 |
| n_bb         | 2: 247 · 3: 241 · 5: 190 (V/tc .230 but 13.8 tc/min) |
| successors   | 2: 237 · 1-2: 189 |
| n_regs       | 2: 249 · 4: 253 · 6: 147 |
| entropy_bits | 4: 170 · 12: 268 · 24: 216 · 31: 196 |
| value_mix    | default: 243 · full64: 197 |

Final policy: n_inputs 10 = 0.77, value_mix default = 0.77, successors 2 = 0.76; all
other knobs still near uniform.

## Conclusions

1. The 3.2x over control is throughput, not better programs: V/tc is lower for the policy.
2. One knob does almost all of it: `-i 10` (cheaper priming; with `-i 50` priming escalates to
   sample size 500 and one control test case took 6 min, slice 35).
3. The reward saturates. The best slices log one violation every 6-12 s, which is the cost of
   confirming and saving a violation, so V/h on v1 tops out around 350-400/h.
4. The control is a weak baseline. A fair comparison needs Revizor with `-i 10` as well.
5. Under CT-SEQ every v1 violation is the same mechanism (one transient access), so "more"
   means re-finding the same class faster.

=> v1 is too easy to show RL > Revizor. Next target: Spectre v4 (rarer per test case, so the
generator distribution should matter).
