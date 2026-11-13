#!/bin/bash
# Tier A queue on this Zen 4 box (Ryzen 7 8700F, SMT off), one campaign after the other, detached:
#   setsid nohup rvzr/SpecRL/genbandit/tier_a/queue_zen4.sh > rvzr/SpecRL/genbandit/tier_a/queue_zen4.out 2>&1 < /dev/null &
# Order: cheap novelty check first, then the campaign with a known positive, then the rare ones.
cd /home/mluo/sca-fuzzer
export PY=/home/mluo/miniconda3/envs/specrl/bin/python INLINE=1
T=rvzr/SpecRL/genbandit/tier_a
echo "queue start $(date)"
ARMS=revizor $T/run_compare.sh zdi 20000      # Zen 4: does 64-bit DIV speculate at all? (Zen 3: no)
$T/run_compare.sh de-dh 10000                  # #DE vs CT-DH: AMD Zen+/Zen 3 violate in 312-785 test cases
$T/run_compare.sh v1var 50000                  # V1-Var under CT-COND: ~3 per 100k on Intel
$T/run_compare.sh sco 50000                    # SCO: ~4 per 500k on Intel, found on Zen 3
echo "queue end $(date)"
