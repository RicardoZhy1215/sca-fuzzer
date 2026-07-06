#!/usr/bin/env python3
"""Evolve a basic Spectre-V1 (bounds-check bypass) Revizor test case with GEPA.

Starting from a seed assembly file, GEPA repeatedly proposes new candidates and
scores each one with Revizor + demo/detect-v1.yaml. The goal is a test case that
reliably triggers a Spectre-V1 contract violation.
"""
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
CFG_V1   = REPO / "demo" / "detect-v1.yaml"

# Revizor prints a "Violations: N" line in its Statistics block; N>0 means a
# contract violation was found. (Older wording "Violation(s) detected/found"
# is also accepted as a fallback.)
VIOLATION_COUNT_RE = re.compile(r"^\s*Violations:\s*(\d+)", re.MULTILINE)
VIOLATION_WORD_RE  = re.compile(r"[Vv]iolation[s]? (detected|found)")


def _detect_violation(out: str) -> bool:
    m = VIOLATION_COUNT_RE.search(out)
    if m:
        return int(m.group(1)) > 0
    return bool(VIOLATION_WORD_RE.search(out))
COND_JMP_RE  = re.compile(r"^\s*(j[a-z]+)\b", re.IGNORECASE | re.MULTILINE)


BACKGROUND = textwrap.dedent("""
    The candidate is a Revizor test case written in Intel syntax without the
    '%' prefix (`.intel_syntax noprefix`). Revizor will assemble it with the
    system `as`, then execute it inside a small sandbox.

    STRUCTURAL RULES (violating any of these => program won't assemble/load)
    --------------------------------------------------------------------
      * Line 1 MUST be:               .intel_syntax noprefix
      * The file MUST contain:        .section .data.main
        near the top, and end with:   .test_case_exit:
      * Local labels like `.l0:` / `.l1:` / `.l2:` may be used as jump targets.
      * Any memory access MUST use r14 as the sandbox base register, e.g.
          mov rax, qword ptr [r14 + rax]
      * Before using a register as an index into [r14 + reg], mask it:
          and reg, 0b111111000000      # keep the index inside the sandbox
        Without the mask the executor will segfault.
      * Do NOT touch rsp, rbp, r14, r15. Safe GPRs to use: rax rbx rcx rdx
        rsi rdi r8 r9 r10 r11 r12 r13.

    GOAL: TRIGGER SPECTRE-V1 (Bounds-Check Bypass)
    ----------------------------------------------
    Intuition: a CONDITIONAL branch guards a memory access. The branch
    predictor may mispredict and let the guarded load issue speculatively
    even when the architectural condition forbids it. The speculatively
    loaded value then drives a second access that leaks through the cache
    side channel.

    A canonical V1 gadget (template -- adapt register names as needed):
        and rax, 0b111111000000          ; reduce entropy of the load index
        and rbx, 0b1000000               ; reduce entropy of the condition
        cmp rbx, 0
        je  .l1                          ; mispredicted branch
        .l0:
            mov rax, qword ptr [r14 + rax]   ; speculative guarded load
        jmp .l2
        .l1:
        .l2:

    HARD CONSTRAINTS for this optimization
    --------------------------------------
      * MUST contain a CONDITIONAL jump (je/jne/jbe/jle/...). The whole point
        of Spectre-V1 is branch misprediction, so a branch-free program cannot
        be a V1 test case.
      * Keep total instructions <= ~40 (otherwise Revizor's measurement
        window may not cover them).
      * Output ONLY the assembly source (no markdown fences, no commentary).
""").strip()


OBJECTIVE = (
    "Refine the seed into a reliable Spectre-V1 (bounds-check bypass) test "
    "case. Output ONLY the full .asm source. Success criteria: (1) Revizor "
    "reports 'Violations detected' with demo/detect-v1.yaml; (2) the asm "
    "contains at least one conditional branch."
)


# ---------------------------------------------------------------------------
# Revizor helpers
# ---------------------------------------------------------------------------
def run_revizor(asm_path: pathlib.Path,
                cfg: pathlib.Path,
                workdir: pathlib.Path,
                n_inputs: int,
                timeout_s: int) -> tuple[bool, bool, str, int, str]:
    """Return (violation_detected, completed, output_tail, returncode, err_head).

    `completed` is True when Revizor finished measurement and printed its
    'Violations: N' statistics line. Note: `reproduce` exits non-zero (rc=1)
    in BOTH the violation and the no-violation case, so rc is NOT a reliable
    success/failure signal -- use `violated`/`completed` instead.
    """
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
    violated = _detect_violation(out)
    # A printed 'Violations: N' line means the candidate assembled and was
    # actually measured (vs. an assembly/load error that aborts before stats).
    completed = bool(VIOLATION_COUNT_RE.search(out))
    # Keep only the tail to save prompt tokens
    tail = "\n".join(out.splitlines()[-60:])
    # Keep a short head excerpt to expose immediate parse/runtime errors.
    head = "\n".join(out.splitlines()[:20])
    return violated, completed, tail, rc, head


def reproduce_config(tmpdir: pathlib.Path) -> pathlib.Path:
    """Clone detect-v1.yaml with the speculation/observation filters OFF.

    Those filters probabilistically *discard* a test case before the full
    contract check when a given measurement happens to see no speculation,
    which makes a real V1 PoC report 'Violations: 0' at random. Turning them
    off makes every input fully evaluated, so a leaking candidate is detected
    deterministically.
    """
    text = CFG_V1.read_text()
    for opt in ("enable_speculation_filter", "enable_observation_filter"):
        if re.search(rf"^\s*{opt}\s*:", text, re.MULTILINE):
            text = re.sub(rf"^\s*{opt}\s*:.*$", f"{opt}: false", text,
                          flags=re.MULTILINE)
        else:
            text += f"\n{opt}: false\n"
    out = tmpdir / "detect-v1-nofilter.yaml"
    out.write_text(text)
    return out


def has_conditional_branch(asm: str) -> bool:
    for m in COND_JMP_RE.finditer(asm):
        op = m.group(1).lower()
        if op != "jmp":
            return True
    return False


# Upper bound on a "reasonable" V1 gadget length; candidates at/above this get
# no size bonus, shorter leaking candidates get progressively more (see below).
MAX_INSTR = 40


def count_instructions(asm: str) -> int:
    """Count real instructions: skip blanks, comments, directives, bare labels.

    A 'label: insn' line (e.g. '.macro.measurement_start: nop ...') still
    counts its trailing instruction.
    """
    n = 0
    for line in asm.splitlines():
        line = line.split("#", 1)[0].strip()      # drop comments
        if not line:
            continue
        if ":" in line:                           # strip a leading label
            line = line.split(":", 1)[1].strip()
        if not line or line.startswith("."):      # bare label or directive
            continue
        n += 1
    return n


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
    check = (example or {}).get("id", "v1")

    # Per-candidate temp dir -> also serves as Revizor's working dir
    tmp_parent = REPO / ".gepa_tmp"
    tmp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(tmp_parent)) as td:
        td = pathlib.Path(td)
        asm_path = _write_asm_to(td, asm)

        if check == "v1":
            # Use detect-v1.yaml with the speculation/observation filters OFF
            # so a leaking candidate is detected deterministically.
            cfg = reproduce_config(td)
            ok, completed, tail, rc, err_head = run_revizor(
                asm_path, cfg, td,
                n_inputs=_STATE["n_inputs"], timeout_s=_STATE["timeout_s"],
            )
            # Scoring is based on the parsed 'Violations: N' count, NOT rc
            # (reproduce exits 1 on both violation and no-violation).
            n_instr = count_instructions(asm)
            if ok:
                # Base 1.0 for a real violation, plus up to +1.0 for being
                # shorter than MAX_INSTR -> GEPA is pushed to MINIMIZE the
                # gadget while keeping it leaking.
                size_bonus = max(0.0, (MAX_INSTR - n_instr) / MAX_INSTR)
                score = 1.0 + size_bonus
                # Surface the detector output that confirms the violation.
                print(f"\n[v1] >>> Violation detected by Revizor "
                      f"({n_instr} instrs, score={score:.3f}); output tail:")
                print(textwrap.indent(tail, "    "))
                print("[v1] <<< end of Revizor output\n")
            elif not completed:
                # Candidate failed to assemble/load -> never measured.
                score = -1.0 if _STATE["strict_eval"] else 0.0
            else:
                score = 0.0                      # assembled & measured, no leak
            _STATE["round_log"].append(
                f"[v1      ] viol={ok}  completed={completed}  "
                f"instrs={n_instr}  score={score:.3f}  rc={rc}"
            )
            return score, {
                "Check":     "detect-v1.yaml (bounds-check bypass), minimize size",
                "Expected":  "'Violations: N' with N>0; fewer instructions = higher score",
                "Violated":  ok,
                "Completed": completed,
                "Instructions": n_instr,
                "ReturnCode": rc,
                "ErrorExcerpt": err_head if not completed else "",
                "RevizorTail": tail,
                "Hint": (
                    "Candidate did NOT assemble/load (no measurement happened). "
                    "Fix the assembly structure first."
                    if not completed else
                    "Assembled and measured fine; now produce a true V1 "
                    "violation (Violations: N must be > 0)."
                    if not ok else
                    f"Good - triggers a V1 violation in {n_instr} instructions. "
                    f"Now make it SHORTER (target < {MAX_INSTR}) while keeping "
                    f"the violation to score higher; remove any instruction not "
                    f"needed for the speculative leak."
                ),
            }

        if check == "has_branch":
            has_br = has_conditional_branch(asm)
            score = 1.0 if has_br else 0.0
            _STATE["round_log"].append(
                f"[hasbranch] has_cond={has_br}"
            )
            return score, {
                "Check":    "Has conditional branch (required for V1)",
                "HasConditionalBranch": has_br,
                "Hint": (
                    "Good - keep this property."
                    if has_br else
                    "If False: ADD a conditional branch (cmp + je/jne/...) that "
                    "guards the speculative load. Spectre-V1 needs misprediction."
                ),
            }

        return 0.0, {"Error": f"unknown check {check}"}


CHECKS = [
    {"id": "v1"},
    {"id": "has_branch"},
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def smoke_test(seed_asm_path: pathlib.Path, n_inputs: int, timeout_s: int):
    """Run all checks on the seed asm without invoking the LLM."""
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
        description="Evolve a basic Spectre-V1 test case with GEPA + Revizor",
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
    p.add_argument("--inputs", type=int, default=100,
                   help="Number of inputs passed to each Revizor run. With the "
                        "speculation filter ON, more inputs give it more chances "
                        "to observe the leak before discarding the test case.")
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-Revizor-run timeout in seconds.")
    p.add_argument("--minibatch", type=int, default=2,
                   help="How many checks the LLM sees per reflection step.")
    p.add_argument("--reflection-lm", type=str, default="ollama/qwen3-coder:30b",
                   help="LiteLLM model name for the proposer.")
    p.add_argument("--strict-eval", action=argparse.BooleanOptionalAction, default=True,
                   help="Penalize candidates when Revizor returns non-zero (rc!=0).")
    p.add_argument("--output", type=pathlib.Path,
                   default=REPO / "gepa_v1_best.asm",
                   help="Where to write the best candidate.")
    return p.parse_args()


def preflight() -> None:
    if not RVZR.exists():
        sys.exit(f"[fatal] {RVZR} not found")
    if not ISA_SPEC.exists():
        sys.exit(f"[fatal] {ISA_SPEC} not found - run: "
                 f"./revizor.py download_spec -a x86-64 --outfile base.json")
    if not CFG_V1.exists():
        sys.exit(f"[fatal] {CFG_V1} not found")
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
