#!/bin/bash
# Leak patterns of all v4 and v1 violations (model only; the executor is asked for base addresses once).
cd /home/mluo/sca-fuzzer
PY=/home/mluo/miniconda3/envs/specrl/bin/python
D=rvzr/SpecRL/genbandit/analysis/leak-patterns
rm -f $D/v4.tsv $D/v1.tsv
$PY rvzr/SpecRL/genbandit/leak_patterns.py v4 $D/v4.tsv > $D/v4.log 2>&1
$PY rvzr/SpecRL/genbandit/leak_patterns.py v1 $D/v1.tsv > $D/v1.log 2>&1
echo done > $D/DONE
