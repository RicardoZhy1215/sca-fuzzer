# GRU vs Transformer ctxrl A/B at Revizor's default v4 knobs + 64 instructions (6 h)

Config: v4.yaml knobs (mem 0.5, -i 50, binary/flags/semaphore/locks on, entropy 10) but
program_size 64; rule2 modes, lineage reward, both arms from Revizor's prior, alternating test
cases. Revizor baseline at the same setting: runs/ttk-default64-261107 (11.3 V/1000, ~73 V/h).

| | new V / 1000 programs | 2nd half | V/h |
|---|---|---|---|
| Revizor (ttk-default64) | 11.3 | | ~73 |
| GRU | 18.2 (15.9-20.5) | 22.1 (1.96x) | 75.9 |
| Transformer | 16.5 (14.3-18.7) | 18.8 (1.67x) | 74.6 |
tf/gru 0.91x (p 0.35). V/h saturated: ~40 s confirmation per violation at -i 50.

## What the policies write differently (analyze_patterns.py, 3000 fresh programs each)
| per program | Revizor | GRU | TF | AUC on hardware (22449 fresh programs) |
|---|---|---|---|---|
| load result used as an address (ld_to_addr) | 9.17 | 12.86 (x1.40) | 11.72 (x1.28) | 0.636: 7.5 -> 24.2 V/1000 bottom -> top quartile |
| loads | 29.5 | 33.6 (x1.14) | 32.8 (x1.11) | 0.621 |
| lock-prefixed RMW stores | 6.76 | 7.65 (x1.13) | 8.40 (x1.24) | 0.561 |
| DATAXFER (mov/movzx/movsx) | 6.42 | 7.29 (x1.14) | 8.40 (x1.31) | 0.565 |
| slow ops (imul...) | 3.83 | 3.18 (x0.83) | 3.89 | 0.474 (negative) |
| dependency depth | 11.7 | 10.8 | 11.1 | 0.459 (negative) |
| store->load pairs via same address register | 7.07 | 8.23 | 8.53 | 0.503 (not predictive) |
The policies moved along the features that predict violations on the hardware; the strongest
is loaded data used as an address (the transmitter of a store-bypass gadget). Same direction as
progrl on the tuned config (ld_to_addr x1.19, loads 37 -> 42%). Correlational; n_load and
ld_to_addr are entangled.
