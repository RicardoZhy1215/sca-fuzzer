#!/bin/bash
# Queue one unmodified `revizor.py fuzz` run of a config: copies the yaml into a new run directory,
# then waits for the executor (a lock shared by all queued runs + any revizor/tcenv already running)
# and runs detached (setsid nohup), so closing the session does not kill it. Runs execute one at a
# time, in the order they got the lock.
#
#   rvzr/SpecRL/genbandit/spectre_v1/queue_revizor.sh <config.yaml> <n programs> [-i inputs (50)] [name]
#   e.g. queue_revizor.sh rvzr/SpecRL/genbandit/spectre_v1/configs/v1-4bb.yaml 3000
set -e
CFG=$(readlink -f "$1"); N=$2; I=${3:-50}
NAME=${4:-$(basename "$CFG" .yaml)}
HERE=$(dirname "$(readlink -f "$0")")
R=$HERE/runs/revizor-$NAME-n$N-i$I-$(date +%y%m%d-%H%M)
mkdir -p "$R"
cp "$CFG" "$R/config.yaml"
cat > "$R/launch.sh" <<EOS
#!/bin/bash
# unmodified Revizor; config copied from $CFG at queue time
cd /home/mluo/sca-fuzzer
echo "queued \$(date)" > $R/queue.log
exec 9>$HERE/.executor.lock
flock 9
# a python process running revizor / tcenv / ttk_bench (not any shell whose text mentions them)
while pgrep -f "^[^ ]*python[^ ]* ([^ ]*/)?(revizor\\.py fuzz|tcenv\\.py|ttk_bench\\.py)" > /dev/null; do sleep 30; done
echo "start \$(date)" >> $R/queue.log
/home/mluo/miniconda3/envs/specrl/bin/python revizor.py fuzz -s base.json -c $R/config.yaml -I $R -n $N -i $I -w $R --nonstop > $R/fuzz.log 2>&1
echo "end \$(date) rc \$?" >> $R/queue.log
/home/mluo/miniconda3/envs/specrl/bin/python $HERE/revizor_curve.py $R > $R/curve.txt 2>&1
EOS
chmod +x "$R/launch.sh"
cd "$R" && setsid nohup ./launch.sh > launch.out 2>&1 < /dev/null &
echo "queued: $R"
