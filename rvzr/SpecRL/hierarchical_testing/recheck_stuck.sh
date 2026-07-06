#!/usr/bin/env bash
# Re-run ONLY the folders whose ct+bpas slow path crashed/timed out, with a big
# timeout. Resolves the deepcopy-timeout cases (they reach priming and yield a
# real NO/YES). The two 'no more checkpoints' assert-crashes will still fail --
# that is a genuine bug in the bpas speculator's taint-tracker rollback.
#
# Run AFTER the main reproduce_ct_bpas.sh has finished (single executor).
set -u
TIMEOUT="${TIMEOUT:-1800}"   # 30 min each; deepcopy-heavy taint states are slow
export TIMEOUT
HERE="/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing"

STUCK=(
  violation-260613-122102   # ASSERT-CRASH (no more checkpoints)
  violation-260613-122217   # ASSERT-CRASH (no more checkpoints)
  violation-260613-122333   # deepcopy timeout
  violation-260613-122448   # deepcopy timeout
  violation-260613-122603   # deepcopy timeout
  violation-260613-160820   # deepcopy timeout
  violation-260613-160934   # deepcopy timeout
)

for f in "${STUCK[@]}"; do
  echo "===== $f (TIMEOUT=${TIMEOUT}s) ====="
  "$HERE/reproduce_ct_bpas.sh" "$f"
done
echo
echo "Re-scan all to fold these back into the summary:"
"$HERE/rescan_ct_bpas.sh"
