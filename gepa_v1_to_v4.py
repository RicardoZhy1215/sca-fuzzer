from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any

REPO     = pathlib.Path(__file__).resolve().parent
RVZR     = REPO / "revizor.py"
ISA_SPEC = REPO / "base.json"
CFG_V4   = REPO / "demo" / "detect-v4.yaml"
CFG_V1   = REPO / "demo" / "detect-v1.yaml"

VIOLATION_RE = re.compile(r"[Vv]iolation[s]? (detected|found)")
COND_JMP_RE  = re.compile(r"^\s*(j[a-z]+)\b", re.IGNORECASE | re.MULTILINE)


BACKGROUND = textwrap.dedent("""
    The candidate is a Revizor test case written in Intel syntax without the
    '%' prefix (`.intel_syntax noprefix`). Revizor will assemble it with the
    system `as`, then execute it inside a small sandbox.

    STRUCTURAL RULES (violating any of these => program won't assemble/load)
    --------------------------------------------------------------------
      * Line 1 MUST be:               .intel_syntax noprefix
      * The file MUST contain, in this order:
            .section .data.main
            .function_0:
            .macro.measurement_start: nop qword ptr [rax + 0xff]
            <body instructions>
            .exit_0:
            .macro.measurement_end:   nop qword ptr [rax + 0xff]
            .section .data.main
            .test_case_exit:nop
      * Basic-block labels like `.bb_0.0:` / `.bb_0.1:` may be used as jump
        targets if needed, but conditional branches are FORBIDDEN here (we
        want a V4-only test case, see below).
      * Any memory access MUST use r14 as the sandbox base register, e.g.
          mov qword ptr [r14 + rdx], rax
      * Before using a register as an index into [r14 + reg], mask it:
          and reg, 0b1111111111111      # byte-granularity (13-bit)
          and reg, 0b1111111111000      # 8-byte-aligned   (13-bit, bot 3 zero)
        Without the mask the executor will segfault.
      * Do NOT touch rsp, rbp, r14, r15. Safe GPRs to use: rax rbx rcx rdx
        rsi rdi r8 r9 r10 r11 r12 r13.

    GOAL: TRIGGER SPECTRE-V4 (Speculative Store Bypass / SSB)
    ---------------------------------------------------------
    Intuition: a STORE to memory may not yet have retired when a later LOAD
    from the SAME (or aliasing) address issues. The store-to-load forwarding
    predictor may let the load *bypass* the pending store and read the STALE
    value, speculatively. If that stale value then drives a cache access, it
    leaks through the cache side channel.

    A canonical V4 gadget (template -- adapt register names as needed):
        and rdx, 0b1111111111000                ; align to 8B
        mov qword ptr [r14 + rdx], 0            ; Store A: "safe" value
        ; enough dependent micro-ops between store and load so the store
        ; stays in-flight (e.g. a long-latency imul chain on rdx):
        imul rdx, rdx, 0xdeadbeef
        imul rdx, rdx, 0x1
        and  rdx, 0b1111111111000               ; re-mask (still same addr)
        mov  rax, qword ptr [r14 + rdx]         ; Load B: may bypass Store A
        and  rax, 0b1111111111111               ; turn leaked value into idx
        mov  rbx, qword ptr [r14 + rax]         ; Transmit: cache side channel

    HARD CONSTRAINTS for this optimization
    --------------------------------------
      * NO conditional jumps (`je/jne/jbe/jle/...`). An unconditional `jmp`
        is fine. This rules out Spectre-V1 and V5 speculation paths, so any
        contract violation must come from SSB.
      * Keep total instructions <= ~40 (otherwise Revizor's measurement
        window may not cover them).
      * Output ONLY the assembly source (no markdown fences, no commentary).
""").strip()


OBJECTIVE = (
    "Transform the seed Spectre-V1 test case into a pure Spectre-V4 "
    "(Speculative Store Bypass) test case. Output ONLY the full .asm source. "
    "Success criteria: (1) Revizor reports 'Violations detected' with "
    "demo/detect-v4.yaml; (2) the asm contains no conditional branch; "
    "(3) enabling x86_executor_enable_ssbp_patch makes the violation vanish."
)


# ---------------------------------------------------------------------------
# Revizor helpers
# ---------------------------------------------------------------------------
def run_revizor(asm_path: pathlib.Path,
                cfg: pathlib.Path,
                workdir: pathlib.Path,
                n_inputs: int,
                timeout_s: int) -> tuple[bool, str, int, str]:
    """Return (violation_detected, output_tail, returncode, error_excerpt)."""
    cmd = [
        sys.executable, str(RVZR), "reproduce",
        "-s", str(ISA_SPEC),
        "-c", str(cfg),
        "-t", str(asm_path),
        "-n", str(n_inputs),
    ]
    try:
        p = subprocess.run(
            cmd,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        rc = p.returncode
        out = (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = -1
        out = f"[TIMEOUT after {timeout_s}s]\n{e.stdout or ''}\n{e.stderr or ''}"
    violated = bool(VIOLATION_RE.search(out))
    # Keep only the tail to save prompt tokens
    tail = "\n".join(out.splitlines()[-60:])
    # Keep a short head excerpt to expose immediate parse/runtime errors.
    head = "\n".join(out.splitlines()[:20])
    return violated, tail, rc, head


def patched_v4_config(tmpdir: pathlib.Path) -> pathlib.Path:
    """Clone detect-v4.yaml but with the SSBP mitigation turned ON."""
    text = CFG_V4.read_text()
    if re.search(r"^\s*x86_executor_enable_ssbp_patch\s*:", text, re.MULTILINE):
        text = re.sub(
            r"^\s*x86_executor_enable_ssbp_patch\s*:.*$",
            "x86_executor_enable_ssbp_patch: true",
            text,
            flags=re.MULTILINE,
        )
    else:
        text += "\nx86_executor_enable_ssbp_patch: true\n"
    out = tmpdir / "detect-v4-ssbp.yaml"
    out.write_text(text)
    return out


def has_conditional_branch(asm: str) -> bool:
    for m in COND_JMP_RE.finditer(asm):
        op = m.group(1).lower()
        if op != "jmp":
            return True
    return False


# ---------------------------------------------------------------------------
# Evaluator (called by GEPA for every (candidate, check))
# ---------------------------------------------------------------------------
# NOTE: We pick these arguments up from module-level state that `main()` sets,
# so the evaluator signature stays compatible with `optimize_anything`.
_STATE: dict[str, Any] = {
    "n_inputs":    50,
    "timeout_s":   120,
    "round_log":   [],
    "strict_eval": True,
}


def _write_asm_to(tmpdir: pathlib.Path, asm: str) -> pathlib.Path:
    asm_path = tmpdir / "cand.asm"
    asm_path.write_text(asm)
    return asm_path


def evaluate(candidate, example=None, **_kwargs):
    """Score a candidate on a single check (from CHECKS)."""
    asm = candidate["asm"] if isinstance(candidate, dict) else candidate
    check = (example or {}).get("id", "v4")

    # Per-candidate temp dir -> also serves as Revizor's working dir
    tmp_parent = REPO / ".gepa_tmp"
    tmp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(tmp_parent)) as td:
        td = pathlib.Path(td)
        asm_path = _write_asm_to(td, asm)

        if check == "v4":
            ok, tail, rc, err_head = run_revizor(
                asm_path, CFG_V4, td,
                n_inputs=_STATE["n_inputs"], timeout_s=_STATE["timeout_s"],
            )
            runtime_ok = (rc == 0)
            if _STATE["strict_eval"] and not runtime_ok:
                score = -1.0
            else:
                score = 1.0 if ok else 0.0
            _STATE["round_log"].append(
                f"[v4      ] viol={ok}  rc={rc}"
            )
            return score, {
                "Check":     "detect-v4.yaml (ct + seq, no branches)",
                "Expected":  "'Violations detected' in output",
                "Violated":  ok,
                "RuntimeOK": runtime_ok,
                "ReturnCode": rc,
                "ErrorExcerpt": err_head if not runtime_ok else "",
                "RevizorTail": tail,
                "Hint": (
                    "Fix assembly/runtime errors first; candidates with rc!=0 "
                    "are penalized under strict evaluation."
                    if not runtime_ok else
                    "Runtime passed; now focus on producing a true V4 violation."
                ),
            }

        if check == "no_branch":
            has_br = has_conditional_branch(asm)
            score = 0.0 if has_br else 1.0
            _STATE["round_log"].append(
                f"[nobranch] has_cond={has_br}"
            )
            return score, {
                "Check":    "No conditional branch (rules out V1)",
                "HasConditionalBranch": has_br,
                "Hint": (
                    "If True: REPLACE all conditional branches with branch-free "
                    "equivalents (cmov, setcc + and/or)."
                    if has_br else
                    "Good - keep this property."
                ),
            }

        if check == "ssbp_kills":
            patched = patched_v4_config(td)
            ok, tail, rc, err_head = run_revizor(
                asm_path, patched, td,
                n_inputs=_STATE["n_inputs"], timeout_s=_STATE["timeout_s"],
            )
            runtime_ok = (rc == 0)
            if _STATE["strict_eval"] and not runtime_ok:
                score = -1.0
            else:
                # We WANT this one to NOT violate -> score inverted
                score = 1.0 if not ok else 0.0
            _STATE["round_log"].append(
                f"[ssbp    ] still_viol={ok}  rc={rc}"
            )
            return score, {
                "Check": "With SSBP mitigation ON, V4 leak should disappear",
                "StillViolatesUnderSSBP": ok,
                "RuntimeOK": runtime_ok,
                "ReturnCode": rc,
                "ErrorExcerpt": err_head if not runtime_ok else "",
                "RevizorTail": tail,
                "Hint": (
                    "Fix assembly/runtime errors first; candidates with rc!=0 "
                    "are penalized under strict evaluation."
                    if not runtime_ok else
                    "If StillViolatesUnderSSBP is True, the leak is NOT from "
                    "store-bypass; try a more canonical store-then-load "
                    "sequence to the same masked [r14+idx]."
                ),
            }

        return 0.0, {"Error": f"unknown check {check}"}


CHECKS = [
    {"id": "v4"},
    {"id": "no_branch"},
    {"id": "ssbp_kills"},
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def smoke_test(seed_asm_path: pathlib.Path, n_inputs: int, timeout_s: int):
    """Run all three checks on the seed asm without invoking the LLM."""
    print(f"[smoke] seed = {seed_asm_path}")
    _STATE["n_inputs"]  = n_inputs
    _STATE["timeout_s"] = timeout_s
    asm = seed_asm_path.read_text()
    for check in CHECKS:
        t0 = time.time()
        score, info = evaluate({"asm": asm}, check)
        print(f"[smoke] check={check['id']:<10} score={score:.2f}  "
              f"({time.time()-t0:.1f}s)")
        for k, v in info.items():
            if k == "RevizorTail":
                continue
            print(f"          {k}: {v}")


def run_optimization(seed_asm_path: pathlib.Path,
                     budget: int,
                     n_inputs: int,
                     timeout_s: int,
                     reflection_lm: str,
                     minibatch: int,
                     strict_eval: bool,
                     output_path: pathlib.Path) -> None:
    # Import lazily so --smoke works without gepa installed.
    from gepa.optimize_anything import (
        optimize_anything,
        GEPAConfig,
        EngineConfig,
        ReflectionConfig,
    )

    _STATE["n_inputs"]  = n_inputs
    _STATE["timeout_s"] = timeout_s
    _STATE["strict_eval"] = strict_eval

    seed_asm = seed_asm_path.read_text()
    print(f"[gepa] seed_asm_path   = {seed_asm_path}")
    print(f"[gepa] budget          = {budget}")
    print(f"[gepa] inputs/run      = {n_inputs}")
    print(f"[gepa] reflection_lm   = {reflection_lm}")
    print(f"[gepa] minibatch       = {minibatch}")
    print(f"[gepa] strict_eval     = {strict_eval}")

    result = optimize_anything(
        seed_candidate={"asm": seed_asm},
        evaluator=evaluate,
        dataset=CHECKS,
        objective=OBJECTIVE,
        background=BACKGROUND,
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=budget),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                reflection_minibatch_size=minibatch,
            ),
        ),
    )

    best_asm = result.best_candidate["asm"]
    output_path.write_text(best_asm)
    print(f"\n[gepa] wrote best candidate -> {output_path}")

    # Dump a few recent round logs for debugging
    tail = _STATE["round_log"][-20:]
    if tail:
        print("[gepa] last rounds:")
        for line in tail:
            print("       " + line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V1->V4 evolution with GEPA + Revizor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("seed_asm", type=pathlib.Path,
                   help="Path to the seed V1 assembly file.")
    p.add_argument("--smoke", action="store_true",
                   help="Only run Revizor checks on the seed, no LLM.")
    p.add_argument("--quick", action="store_true",
                   help="Use a low-cost preset for quick local-model probing.")
    p.add_argument("--budget", type=int, default=60,
                   help="Total number of evaluator invocations.")
    p.add_argument("--inputs", type=int, default=50,
                   help="Number of inputs passed to each Revizor run.")
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-Revizor-run timeout in seconds.")
    p.add_argument("--minibatch", type=int, default=2,
                   help="How many checks the LLM sees per reflection step.")
    p.add_argument("--reflection-lm", type=str, default="ollama/qwen2.5-coder:7b",
                   help="LiteLLM model name for the proposer.")
    p.add_argument("--strict-eval", action=argparse.BooleanOptionalAction, default=True,
                   help="Penalize candidates when Revizor returns non-zero (rc!=0).")
    p.add_argument("--output", type=pathlib.Path,
                   default=REPO / "gepa_v4_best.asm",
                   help="Where to write the best candidate.")
    return p.parse_args()


def preflight() -> None:
    if not RVZR.exists():
        sys.exit(f"[fatal] {RVZR} not found")
    if not ISA_SPEC.exists():
        sys.exit(f"[fatal] {ISA_SPEC} not found - run: "
                 f"./revizor.py download_spec -a x86-64 --outfile base.json")
    if not CFG_V4.exists():
        sys.exit(f"[fatal] {CFG_V4} not found")
    if shutil.which("as") is None:
        print("[warn] GNU `as` not on PATH - Revizor may fail to assemble "
              "candidates.")


def main() -> None:
    args = parse_args()
    if not args.seed_asm.exists():
        sys.exit(f"[fatal] seed not found: {args.seed_asm}")
    preflight()

    if args.smoke:
        smoke_test(args.seed_asm, args.inputs, args.timeout)
        return

    if args.quick:
        # Keep local open-source runs cheap and fast for first-pass viability checks.
        args.budget = min(args.budget, 24)
        args.inputs = min(args.inputs, 20)
        args.timeout = min(args.timeout, 45)
        args.minibatch = min(args.minibatch, 1)
        print("[quick] enabled: budget<=24, inputs<=20, timeout<=45s, minibatch=1")

    run_optimization(
        seed_asm_path=args.seed_asm,
        budget=args.budget,
        n_inputs=args.inputs,
        timeout_s=args.timeout,
        reflection_lm=args.reflection_lm,
        minibatch=args.minibatch,
        strict_eval=args.strict_eval,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
