#!/bin/bash
# contract + mitigation classification of every v4 and v1 violation behind the paper's figures
# (classify_contract.py; resumable: rows already in the tsv are skipped)
cd /home/mluo/sca-fuzzer
G=rvzr/SpecRL/genbandit
PY=/home/mluo/miniconda3/envs/specrl/bin/python
LOG=$G/analysis/v4-contract/progress.log
exec 9>$G/spectre_v1/.executor.lock
flock 9
while pgrep -f "^[^ ]*python[^ ]* ([^ ]*/)?(revizor\.py fuzz|tcenv\.py|ttk_bench\.py|time_breakdown\.py)" > /dev/null; do sleep 30; done
echo "v4 start $(date)" >> $LOG
$PY -u $G/classify_contract.py v4 $G/analysis/v4-contract/classes.tsv > $G/analysis/v4-contract/run.log 2>&1
echo "v4 end $(date) rc $?" >> $LOG
echo "v1 start $(date)" >> $LOG
$PY -u $G/classify_contract.py v1 $G/spectre_v1/analysis/contract/classes.tsv > $G/spectre_v1/analysis/contract/run.log 2>&1
echo "v1 end $(date) rc $?" >> $LOG
