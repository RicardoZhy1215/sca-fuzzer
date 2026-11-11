#!/bin/bash
# Every violation of the given runs through: triage (real? FPVI family?) -> minimize -> classify
# (roles, secret, mechanism). Skips steps already done for a violation. Detached; the executor
# lock serializes it with other campaigns.
#   rvzr/SpecRL/genbandit/sse_fp/analyze_all.sh <out-dir> <run-dir>...
set -u
ROOT=$(git rev-parse --show-toplevel)
S=$ROOT/rvzr/SpecRL/genbandit/sse_fp
PY=$(readlink -f "${PY:-$(command -v python)}")
mkdir -p "$1"; OUT=$(readlink -f "$1"); shift
RUNS=$(for r in "$@"; do readlink -f "$r"; done | tr '\n' ' ')
cat > "$OUT/launch.sh" <<EOS
#!/bin/bash
cd $ROOT
exec 9>$S/../spectre_v1/.executor.lock
flock 9
while pgrep -f "^[^ ]*python[^ ]* ([^ ]*/)?(revizor\.py fuzz|tcenv\.py|ttk_bench\.py)" > /dev/null; do sleep 30; done
V=\$(for r in $RUNS; do ls -d \$r/violation-*; done)
T=\$(for v in \$V; do [ -f \$v/triage/verdict.txt ] || echo \$v; done)
echo "triage \$(echo \$T | wc -w) start \$(date)" >> $OUT/progress.log
[ -n "\$T" ] && $PY $S/triage.py \$T >> $OUT/triage.log 2>/dev/null
M=\$(for v in \$V; do [ -f \$v/minimize/min.asm ] || echo \$v; done)
echo "minimize \$(echo \$M | wc -w) start \$(date)" >> $OUT/progress.log
[ -n "\$M" ] && $PY $S/minimize.py \$M >> $OUT/minimize.log 2>/dev/null
echo "classify \$(echo \$V | wc -w) start \$(date)" >> $OUT/progress.log
rm -f $OUT/classes.tsv
$PY $S/classify.py \$V --out $OUT/classes.tsv > /dev/null 2>&1
cat \$(for v in \$V; do echo \$v/triage/verdict.txt; done) > $OUT/verdicts.tsv
echo "done \$(date)" >> $OUT/progress.log
EOS
chmod +x "$OUT/launch.sh"
(cd "$OUT" && setsid nohup ./launch.sh > launch.out 2>&1 < /dev/null &)
echo "started: $OUT"
