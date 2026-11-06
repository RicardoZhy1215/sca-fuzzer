# progrl on v4 -- 12 h (2026-11-05 06:14-18:30)

`tcenv.py --mode bypass --policy progrl --minutes 720`, config = coupled incumbent (64 instr, mem
0.75, n_regs 6, entropy 16, -i 10, binary/flags/semaphore off, locks on, spec filter on).
lr 3e-4, KL 0.1, SIL 1.0, update every 256 results (459 updates, 0.87 s each on cores 4-7).
Files: analysis.txt (analyze_progrl.py), inspect.txt (inspect_progrl.py).

## Per-program violation rate (the learning signal)
| | tc | V | V/1000 tc | V/h | tc/min |
|---|---|---|---|---|---|
| same config, Revizor's distribution (coupled run, live) | 108006 | 2281 | 21.1 | 283 | 223 |
| progrl hours 0-2 | 23765 | 575 | 24.2 | 287 | 198 |
| progrl hours 2-4 / 4-6 / 6-8 | | | 28.5 / 31.8 / 30.6 | 295 / 297 / 293 | |
| progrl hours 8-12 | 35470 | 1185 | 33.4 | 296 | 148 |
Last 4 h vs Revizor's distribution: x1.58 violations per program (Poisson SE ~3% each).
V/h flat (saturated: confirmation time dominates; tc/min fell 198 -> 148 as V per program rose).
NOT yet an interleaved A/B (the 21.1 is from another day / process).

## What the policy changed (2000 sampled programs each)
- loads 37.3% -> 42.1% of instructions; stores 37.7 -> 35.1; non-memory 25.0 -> 22.8
- load destination later used as an address (within 8): 10.3 -> 12.3 per program (x1.19)
- store->load through the same address register: unchanged (x1.00-1.03)
- up: xor/and REG,REG, bswap, lea, movzx/mov REG,MEM; down (x0.1-0.35): mov REG,IMM, mov REG,REG,
  or/test REG,REG, and REG,IMM
- KL to prior per decision: instruction 0.099, register 0.035, address register 0.031 nats;
  entropy 3.19 -> 3.10; 3523 distinct positive programs (no duplicates, no collapse)
Hypothesis (unverified): it favors address dependence on loaded data (late store-address
resolution) and avoids instructions that overwrite registers with input-independent values.
