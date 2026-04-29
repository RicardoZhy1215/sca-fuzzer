#!/usr/bin/env bash
# Run hi_Specenv_check.py on all violation_20260423_*.asm under debug_asm/
# (excludes *patched* derivatives). From hierarchical_testing:
#   ./run_violation_20260423_asms.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ASM_DIR="${SCRIPT_DIR}/debug_asm"
failed=0
count=0
while IFS= read -r -d '' f; do
  rel="./debug_asm/$(basename "$f")"
  echo "========================================================================"
  echo "==> python hi_Specenv_check.py --asm $rel"
  if ! python hi_Specenv_check.py --asm "$rel"; then
    echo "FAILED: $rel" >&2
    ((failed++)) || true
  fi
  ((count++)) || true
done < <(find "$ASM_DIR" -maxdepth 1 -type f -name 'violation_20260423_*.asm' ! -name '*.patched.asm' -print0 2>/dev/null | sort -z -V)
if ((count == 0)); then
  echo "No files matched: debug_asm/violation_20260423_*.asm (non-patched)" >&2
  exit 0
fi
echo "========================================================================"
if ((failed > 0)); then
  echo "Done: $count run(s), $failed failed." >&2
  exit 1
fi
echo "Done: $count run(s), all succeeded."
