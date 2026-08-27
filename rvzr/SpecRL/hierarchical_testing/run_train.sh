#!/usr/bin/env bash
# ==============================================================================
# Launch a SpecRL training campaign (wandb-enabled).
# ==============================================================================
# Usage:
#   ./run_train.sh v1                          # Spectre V1 reproduction
#   ./run_train.sh v4                          # Spectre V4 reproduction
#   ./run_train.sh v1 --specrl-wandb-run foo   # extra args pass through to train.py
#   ./run_train.sh <path/to/other.yaml>        # any campaign config
#
# Environment overrides:
#   RUN_NAME=...      wandb run name          (default: <campaign>-<timestamp>)
#   PROJECT=...       wandb project           (default: SpecRL-hierarchical)
#   PY=...            python interpreter      (default: the `specrl` conda env)
#   NO_WANDB=1        disable wandb logging
#   WANDB_MODE=offline   log locally, sync later with `wandb sync`
#   FORCE=1           skip the "executor busy" guard (see below)
#   DRY_RUN=1         run the pre-flight checks only, then stop
#
# Run in the background (training takes hours; the log is written either way):
#   nohup ./run_train.sh v1 >/dev/null 2>&1 &
#   tail -f logs/<campaign>-<timestamp>.log
#
# !! The rvzr executor kernel module is a SINGLE GLOBAL RESOURCE. One trainer OR
#    one `rvzr fuzz` campaign at a time -- overlapping runs interleave their
#    test-case loads and corrupt each other's hardware traces. The pre-flight
#    below refuses to start if something else is already using it.
# ==============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PY:-/home/mluo/miniconda3/envs/specrl/bin/python}"
PROJECT="${PROJECT:-SpecRL-hierarchical}"
LOG_DIR="$HERE/logs"

die() { echo "ERROR: $*" >&2; exit 2; }

# ------------------------------------------------------------------ campaign --
[ $# -ge 1 ] || die "usage: $(basename "$0") {v1|v4|<config.yaml>} [extra train.py args...]"
CAMPAIGN="$1"; shift
case "$CAMPAIGN" in
    v1|v4) CONFIG="$HERE/${CAMPAIGN}.yaml" ;;
    *.yaml) CONFIG="$(cd "$(dirname "$CAMPAIGN")" && pwd)/$(basename "$CAMPAIGN")"
            CAMPAIGN="$(basename "$CONFIG" .yaml)" ;;
    *) die "unknown campaign '$CAMPAIGN' (expected v1, v4, or a path to a .yaml)" ;;
esac
STAMP="$(date +%y%m%d-%H%M%S)"
RUN_NAME="${RUN_NAME:-${CAMPAIGN}-${STAMP}}"
LOG_FILE="$LOG_DIR/${CAMPAIGN}-${STAMP}.log"

# ----------------------------------------------------------------- preflight --
echo "== pre-flight =================================================="

[ -f "$CONFIG" ]           || die "config not found: $CONFIG"
[ -f "$REPO/base.json" ]   || die "missing $REPO/base.json (run: rvzr download_spec)"
[ -x "$PY" ]               || die "python not found: $PY   (override with PY=...)"
[ -e /sys/rvzr_executor ]  || die "executor module not loaded -- sudo insmod $REPO/rvzr/executor_km/rvzr_executor.ko"
# A COPY of rvzr is installed in this env's site-packages and it does NOT have
# the SpecRL additions (create_test_case_SpecRL, ...). train.py's sys.path.insert
# fixes the driver but NOT the Ray rollout workers, which start from a fresh
# interpreter and fall back to site-packages ->
#     AttributeError: 'X86Generator' object has no attribute 'create_test_case_SpecRL'
# PYTHONPATH is inherited by every ray worker and is searched BEFORE
# site-packages, so exporting it here fixes driver and workers alike.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "  config      : $CONFIG"
echo "  python      : $PY"
echo "  executor    : loaded (/sys/rvzr_executor)"

# The executor is exclusive: refuse to start alongside another user of it.
BUSY="$(pgrep -af 'train\.py|revizor\.py|rvzr (fuzz|tfuzz|minimize)' 2>/dev/null \
        | grep -v 'run_train\.sh' | grep -v "^$$ " || true)"
if [ -n "$BUSY" ]; then
    echo "  executor    : BUSY --"
    echo "$BUSY" | sed 's/^/                /'
    if [ "${FORCE:-0}" != "1" ]; then
        die "another job is using the executor; wait for it or re-run with FORCE=1"
    fi
    echo "  (FORCE=1 -- proceeding anyway; traces from both runs may be garbage)"
else
    echo "  contention  : none (no other train.py / rvzr job running)"
fi

# Campaign-specific hardware prerequisite: v4 needs a CPU with an SSB channel.
# NOTE: the *kernel's* system-wide setting is NOT a blocker. The executor module
# writes MSR_IA32_SPEC_CTRL.SSBD directly around each measurement
# (executor_km/x86/special_registers.c: apply_msr_mask(..., enable_ssbp_patch)),
# so `x86_executor_enable_ssbp_patch: false` clears SSBD for the test case even
# when the kernel booted with spec_store_bypass_disable=on. Only a CPU with no
# SSB channel at all ("Not affected") makes v4 unreachable.
SSB_SYSFS=/sys/devices/system/cpu/vulnerabilities/spec_store_bypass
if [ "$CAMPAIGN" = "v4" ] && [ -r "$SSB_SYSFS" ]; then
    SSB="$(cat "$SSB_SYSFS")"
    case "${SSB,,}" in
        *"not affected"*)
            die "CPU reports SSB as 'Not affected' -- Spectre-v4 cannot be reproduced here" ;;
        *)
            echo "  spec_store_bypass: $SSB"
            echo "                (kernel-level only; executor clears SSBD per measurement)" ;;
    esac
fi

# wandb credentials.
WANDB_ARGS=()
if [ "${NO_WANDB:-0}" = "1" ]; then
    WANDB_ARGS=(--specrl-no-wandb)
    echo "  wandb       : disabled (NO_WANDB=1)"
elif [ "${WANDB_MODE:-}" = "offline" ]; then
    echo "  wandb       : offline (sync later: wandb sync $HERE/wandb/offline-run-*)"
else
    if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
        die "wandb not authenticated -- run 'wandb login', or set WANDB_API_KEY, or use NO_WANDB=1 / WANDB_MODE=offline"
    fi
    echo "  wandb       : $PROJECT / $RUN_NAME"
fi

# Echo what the campaign config actually resolves to -- catches a wrong contract
# or a missing branch setting before burning hours of rollouts.
"$PY" - "$CONFIG" "$REPO" <<'PYEOF' || die "pre-flight failed (see above)"
import os, sys, warnings
warnings.filterwarnings("ignore")
import rvzr.arch.x86.generator as _g
from rvzr.config import CONF
from rvzr import factory

# Guard against the stale site-packages copy of rvzr shadowing the checkout.
_repo = sys.argv[2]
if not os.path.abspath(_g.__file__).startswith(os.path.abspath(_repo)) \
        or not hasattr(_g.X86Generator, "create_test_case_SpecRL"):
    print(f"  rvzr        : WRONG COPY -> {_g.__file__}", file=sys.stderr)
    print(f"                expected it under {_repo}; the installed copy lacks "
          f"the SpecRL methods.\n"
          f"                fix: export PYTHONPATH={_repo}  (or reinstall: "
          f"pip install -e {_repo})", file=sys.stderr)
    sys.exit(1)
print(f"  rvzr        : {os.path.dirname(os.path.dirname(os.path.dirname(_g.__file__)))}")

CONF.load(sys.argv[1], "")
policy = "random" if CONF.max_bb_per_function > 1 else "first"
print(f"  contract    : obs={CONF.contract_observation_clause} "
      f"exec={CONF.contract_execution_clause} "
      f"-> {factory._SPECULATORS_X86[factory._get_exec_clause_name()].__name__}")
print(f"  ssbd        : x86_executor_enable_ssbp_patch={CONF.x86_executor_enable_ssbp_patch}")
print(f"  cfg shape   : bb={CONF.min_bb_per_function}..{CONF.max_bb_per_function} "
      f"successors={CONF.min_successors_per_bb}..{CONF.max_successors_per_bb} "
      f"-> insert_bb_policy={policy}")
print(f"  entropy     : {CONF.data_generator_entropy_bits} bits, "
      f"inputs_per_class={CONF.inputs_per_class}")
PYEOF

mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR"
echo "  log         : $LOG_FILE"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "== DRY_RUN=1 -- pre-flight passed, not launching ================"
    exit 0
fi
echo "== launching ==================================================="

# ------------------------------------------------------------------- launch --
cd "$HERE" || die "cannot cd to $HERE"
{
    echo "[$(date '+%F %T')] starting $CAMPAIGN (run=$RUN_NAME)"
    "$PY" train.py \
        --config "$CONFIG" \
        --include-dir "$HERE" \
        --specrl-wandb-project "$PROJECT" \
        --specrl-wandb-run "$RUN_NAME" \
        "${WANDB_ARGS[@]}" \
        "$@"
    rc=$?
    echo "[$(date '+%F %T')] finished $CAMPAIGN (exit=$rc)"
    exit $rc
} 2>&1 | tee -a "$LOG_FILE"
rc=${PIPESTATUS[0]}

# A hard kill (Ctrl-C, OOM) can leave ray workers holding the executor open,
# which then blocks the next run's pre-flight. Clean them up before retrying.
if [ "$rc" -ne 0 ]; then
    echo "run failed (exit=$rc). If ray workers survived, clear them with:"
    echo "    $(dirname "$PY")/ray stop --force"
fi
exit "$rc"
