#!/bin/bash
# Tier A queue on this Zen 4 box (Ryzen 7 8700F, SMT off), one campaign after the other, detached:
#   setsid nohup rvzr/SpecRL/genbandit/tier_a/queue_zen4.sh > rvzr/SpecRL/genbandit/tier_a/queue_zen4.out 2>&1 < /dev/null &
# FROM=<k> starts at item k (e.g. after a restart). Each item waits for the executor lock, so a queue
# started while a campaign runs simply waits for it.
# Order: cheap novelty check first, then the campaign with a known positive, then the rare ones
# (20k programs each: if nothing shows up, so be it).
cd /home/mluo/sca-fuzzer
export PY=/home/mluo/miniconda3/envs/specrl/bin/python INLINE=1
T=rvzr/SpecRL/genbandit/tier_a
FROM=${FROM:-1}
echo "queue start $(date) from item $FROM"
[ $FROM -le 1 ] && ARMS=revizor $T/run_compare.sh zdi 20000   # Zen 4: does 64-bit DIV speculate at all? (Zen 3: no)
[ $FROM -le 2 ] && $T/run_compare.sh de-dh 10000               # #DE vs CT-DH: AMD Zen+/Zen 3 violate in 312-785 test cases
[ $FROM -le 3 ] && $T/run_compare.sh v1var 20000               # V1-Var under CT-COND: ~3 per 100k on Intel
[ $FROM -le 4 ] && $T/run_compare.sh sco 20000                 # SCO: ~4 per 500k on Intel, found on Zen 3
echo "queue end $(date)"
