"""
Spectre-v4 violation diagnostic — scheme C (overview) + scheme A (definitive).

Scheme C (cheap, weak):
  Run the program once, look at end-of-program register snapshot, check
  whether any (store_base, load_base) pair has the same end-state value.
  Finds nothing if the gadget's base registers got overwritten before exit.

Scheme A (per-pair, definitive):
  For each cross-register store→load candidate, generate a patched copy
  of the asm with three injected instructions:
        mov [r14 + 0x2000], <store_base>   ; just before the store
        mov [r14 + 0x2008], <load_base>    ; just before the load
        mov rax, [r14 + 0x2000]            ; right after the load
        mov rbx, [r14 + 0x2008]
        jmp .exit_0
  The early-exit guarantees that no later instruction overwrites the
  saved values. Reading rax / rbx via arch_executor gives the EXACT
  post-instrumentation address the store and the load used.

  Saver slots 0x2000 and 0x2008 live in the kernel module's
  upper_overflow region — outside the 0x0..0x1FFF range that the
  program's `and reg, 0b1111111111111` instrumentation can reach, so
  the program code cannot trample our saved values.

Run:
  cd /home/hz25d/sca-fuzzer/rvzr/SpecRL/hierarchical_testing
  python diagnose_v4_scheme_c.py debug_asm/violation_20260504_204546_898427.asm
  # add --no-scheme-a to skip the per-pair patched runs.
"""
from __future__ import annotations

import argparse
import os
import sys
import re
from collections import Counter
from typing import List, Tuple

import numpy as np

# Ensure rvzr/ is on path before importing config-dependent modules.
_REPO_ROOT = "/home/hz25d/sca-fuzzer"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LOCAL = os.path.dirname(os.path.abspath(__file__))
if _LOCAL not in sys.path:
    sys.path.insert(0, _LOCAL)

from rvzr.config import CONF  # noqa: E402

CONF.load(
    os.path.join(_LOCAL, "config.yaml"),
    os.path.join(_REPO_ROOT, "rvzr"),
)

from rvzr.arch.x86.target_desc import X86TargetDesc  # noqa: E402
from rvzr.arch.x86.generator import X86Generator  # noqa: E402
from rvzr.arch.x86.asm_parser import X86AsmParser  # noqa: E402
from rvzr.elf_parser import ELFParser  # noqa: E402
from rvzr.isa_spec import InstructionSet  # noqa: E402
from rvzr import factory  # noqa: E402
from rvzr.code_generator import assemble  # noqa: E402

# 6-GPR snapshot order, matches RawHTraceSample (trace, pfc0..pfc4)
# and rvzr.traces._REG_ID_TO_NAME_X86.
SNAPSHOT_REGS: List[str] = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi"]


# ----------------------------------------------------------------------------
# 1. Static parse: extract (store|load, base_reg, line_no) tuples.
# ----------------------------------------------------------------------------
# Match `[r14 + reg]` where reg is one of the 6 snapshot GPRs. We don't bother
# with displacements / scale because revizor's sandbox always uses [r14 + reg].
_MEM_RE = re.compile(r"\[\s*r14\s*\+\s*(rax|rbx|rcx|rdx|rsi|rdi)\s*\]", re.IGNORECASE)


def _classify(line: str) -> Tuple[str, str] | None:
    """
    Return (kind, base_reg) where kind in {"store", "load"}, or None if the
    line is not a memory-touching agent instruction.
    """
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("."):
        return None
    if "# instrumentation" in s:
        return None
    s_clean = s.split("#", 1)[0].strip()

    m = _MEM_RE.search(s_clean)
    if not m:
        return None
    base = m.group(1).lower()

    # Heuristic: a memory operand is a store iff it appears as the FIRST
    # operand of a write-style mnemonic. Revizor asm always has the
    # destination first ('mov [r14+reg], src' is a store; 'mov reg, [r14+reg]'
    # is a load).
    head = s_clean.lower()
    # remove the mnemonic prefix and locking
    head = re.sub(r"^(lock\s+)?", "", head)
    mnemonic_match = re.match(r"([a-z]+)\s+(.*)", head)
    if not mnemonic_match:
        return None
    op_text = mnemonic_match.group(2)

    # If the FIRST operand contains [r14+...], it's a store... EXCEPT for
    # single-operand mul/div/imul/idiv/neg/not where the memory operand is
    # actually a source (these are unary ops where the explicit operand is
    # the source and rax/rdx is implicit dest).
    mnemonic = mnemonic_match.group(1)
    UNARY_LOAD_MNEMONICS = {"mul", "div", "imul", "idiv"}
    operands = [o.strip() for o in op_text.split(",")]
    if not operands:
        return None
    if mnemonic in UNARY_LOAD_MNEMONICS and len(operands) == 1:
        return ("load", base)
    if "[r14" in operands[0]:
        return ("store", base)
    return ("load", base)


def _parse_violation_asm(path: str) -> List[Tuple[int, str, str, str]]:
    """
    Return a list of (line_no, kind, base_reg, raw_line) for every
    agent-emitted memory instruction.
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            cls = _classify(raw)
            if cls is None:
                continue
            kind, base = cls
            out.append((line_no, kind, base, raw.rstrip()))
    return out


# ----------------------------------------------------------------------------
# 2. Run via arch_executor and capture per-input register snapshots.
# ----------------------------------------------------------------------------
def _build_test_case(asm_path: str):
    target_desc = X86TargetDesc()
    isa = InstructionSet(os.path.join(_REPO_ROOT, "base.json"))
    asm_parser = X86AsmParser(isa, target_desc)
    elf_parser = ELFParser(target_desc)
    generator = X86Generator(
        seed=CONF.program_generator_seed,
        instruction_set=isa,
        target_desc=target_desc,
        asm_parser=asm_parser,
        elf_parser=elf_parser,
    )
    # parse_file already performs assign_obj + assemble + populate_elf_data.
    program = asm_parser.parse_file(asm_path, generator, elf_parser)
    return program


def _capture_register_state(asm_path: str, num_inputs: int) -> List[List[int]]:
    """
    Returns [[rax, rbx, rcx, rdx, rsi, rdi], ...], one row per input.
    """
    program = _build_test_case(asm_path)

    # Same arch_executor used by SpecEnv: kernel-module GPR mode.
    arch_executor = factory.get_executor(enable_mismatch_check_mode=True)
    arch_executor.valid_mem_base = 0x0
    arch_executor.valid_mem_limit = 0x100000

    input_gen = factory.get_data_generator(CONF.data_generator_seed)
    inputs = input_gen.generate(num_inputs, n_actors=1)

    arch_executor.load_test_case(program)
    traces = arch_executor.trace_test_case(inputs, n_reps=1)

    snapshots = []
    for tr in traces:
        samples = tr.get_raw_readings()
        if len(samples) == 0:
            snapshots.append([0] * 6)
            continue
        s = samples[0]
        snapshots.append([
            int(s["trace"]),  # rax
            int(s["pfc0"]),   # rbx
            int(s["pfc1"]),   # rcx
            int(s["pfc2"]),   # rdx
            int(s["pfc3"]),   # rsi
            int(s["pfc4"]),   # rdi
        ])
    return snapshots


def _equality_encoding(snapshot: List[int]) -> List[int]:
    """Same encoding used by SpecEnv._collect_reg_state."""
    counts = Counter(snapshot)
    return [1 if counts[v] > 1 else 0 for v in snapshot]


# ----------------------------------------------------------------------------
# Scheme A: checkpoint injection for a single (store, load) pair.
# ----------------------------------------------------------------------------
SAVER_OFF_STORE = 0x2000  # in upper_overflow, unreachable by instrumented code
SAVER_OFF_LOAD = 0x2008


def _patch_asm_for_pair(orig_asm: str, store_line: int, store_reg: str,
                        load_line: int, load_reg: str, out_path: str) -> None:
    """
    Write a patched copy of orig_asm with checkpoint instructions injected
    around the (store, load) pair.

    Layout:
      ... unchanged ...
      mov [r14 + 0x2000], <store_base>     <-- INJECTED right before store
      <original store>
      ... unchanged (instrumentation, original loads/stores) ...
      mov [r14 + 0x2008], <load_base>      <-- INJECTED right before load
      <original load>
      ... unchanged ...
    .exit_0:
      mov rax, [r14 + 0x2000]              <-- INJECTED at .exit_0 label
      mov rbx, [r14 + 0x2008]
      ... original .exit_0 body ...

    The saver slots live at offsets 0x2000+ which the program's
    `and reg, 0b1111111111111` instrumentation cannot reach, so the saved
    values survive any subsequent stores in the original program. We don't
    need a `jmp .exit_0` to early-exit (which would break the parser's
    "terminator at end of BB" rule).

    line numbers are 1-based and refer to lines in orig_asm.
    """
    with open(orig_asm, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out: List[str] = []
    for i, line in enumerate(lines, start=1):
        if i == store_line:
            out.append(
                f"mov qword ptr [r14 + {SAVER_OFF_STORE}], {store_reg}  "
                f"# CHECKPOINT save store base ({store_reg})\n"
            )
            out.append(line)
        elif i == load_line:
            out.append(
                f"mov qword ptr [r14 + {SAVER_OFF_LOAD}], {load_reg}  "
                f"# CHECKPOINT save load base ({load_reg})\n"
            )
            out.append(line)
        else:
            out.append(line)
            stripped = line.strip()
            # Recover at .exit_0 — first thing after the label.
            if stripped == ".exit_0:":
                out.append(
                    f"mov rax, qword ptr [r14 + {SAVER_OFF_STORE}]  "
                    f"# CHECKPOINT recover store base into rax\n"
                )
                out.append(
                    f"mov rbx, qword ptr [r14 + {SAVER_OFF_LOAD}]  "
                    f"# CHECKPOINT recover load base into rbx\n"
                )
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out)


def _run_pair_scheme_a(orig_asm: str, store_line: int, store_reg: str,
                       load_line: int, load_reg: str, num_inputs: int,
                       work_dir: str) -> List[Tuple[int, int]] | None:
    """
    Run scheme A on a single pair. Returns a list of (rax, rbx) end-state
    values (one per input), or None if patching/parsing/running fails.
    """
    patched = os.path.join(
        work_dir,
        f"patched_pair_L{store_line}_{store_reg}_to_L{load_line}_{load_reg}.asm",
    )
    try:
        _patch_asm_for_pair(orig_asm, store_line, store_reg,
                            load_line, load_reg, patched)
    except Exception as exc:  # noqa: BLE001
        print(f"  [scheme-A] patch failed: {exc}")
        return None

    try:
        program = _build_test_case(patched)
    except Exception as exc:  # noqa: BLE001
        print(f"  [scheme-A] parse/assemble failed: {exc}")
        return None

    try:
        arch_executor = factory.get_executor(enable_mismatch_check_mode=True)
        arch_executor.valid_mem_base = 0x0
        arch_executor.valid_mem_limit = 0x100000
        input_gen = factory.get_data_generator(CONF.data_generator_seed)
        inputs = input_gen.generate(num_inputs, n_actors=1)

        arch_executor.load_test_case(program)
        traces = arch_executor.trace_test_case(inputs, n_reps=1)
    except Exception as exc:  # noqa: BLE001
        print(f"  [scheme-A] run failed: {exc}")
        return None

    pairs = []
    for tr in traces:
        samples = tr.get_raw_readings()
        if len(samples) == 0:
            pairs.append((0, 0))
            continue
        s = samples[0]
        pairs.append((int(s["trace"]), int(s["pfc0"])))  # rax, rbx
    return pairs


def _run_scheme_a_for_all_cross_pairs(
    orig_asm: str,
    mem_ops: List[Tuple[int, str, str, str]],
    num_inputs: int,
) -> None:
    stores = [(ln, base, raw) for (ln, kind, base, raw) in mem_ops if kind == "store"]
    loads = [(ln, base, raw) for (ln, kind, base, raw) in mem_ops if kind == "load"]

    # Note: we INCLUDE same-register pairs here. Under scheme C they were
    # "trivially equal" at end-state, but under scheme A we capture
    # post-instrumentation addresses at the moment of execution — and the
    # program's `and reg, 0x1FFF` is re-applied before every access, so
    # even same-register pairs are non-trivial v4 candidates.
    candidate_pairs = []
    for (s_line, s_base, s_raw) in stores:
        for (l_line, l_base, l_raw) in loads:
            if l_line <= s_line:
                continue
            candidate_pairs.append((s_line, s_base, s_raw, l_line, l_base, l_raw))

    if not candidate_pairs:
        print("\n=== Scheme A — no cross-register candidate pairs to test ===")
        return

    work_dir = os.path.join(os.path.dirname(os.path.abspath(orig_asm)), "_diag_patched")
    os.makedirs(work_dir, exist_ok=True)

    print(f"\n=== Scheme A — testing {len(candidate_pairs)} cross-register pair(s) ===\n")
    print(f"  saver slots: [r14+0x{SAVER_OFF_STORE:x}] (store), "
          f"[r14+0x{SAVER_OFF_LOAD:x}] (load)")
    print(f"  patched asms cached in: {work_dir}\n")

    pair_results: List[Tuple[int, str, int, str, int, int]] = []  # (s_l, s_r, l_l, l_r, n_match, n_total)
    for (s_line, s_base, s_raw, l_line, l_base, l_raw) in candidate_pairs:
        result = _run_pair_scheme_a(orig_asm, s_line, s_base, l_line, l_base,
                                    num_inputs, work_dir)
        if result is None:
            continue
        n_match = sum(1 for (rax, rbx) in result if rax == rbx)
        n_total = len(result)
        pair_results.append((s_line, s_base, l_line, l_base, n_match, n_total))
        # Only print full per-input detail for pairs with at least one match.
        if n_match > 0:
            print(f"  [pair] L{s_line:>3} store on {s_base}  →  L{l_line:>3} load on {l_base}  "
                  f"({n_match}/{n_total})")
            for i, (rax, rbx) in enumerate(result):
                mark = "✓" if rax == rbx else "·"
                print(f"           input {i}: 0x{rax:>10x} | 0x{rbx:>10x}  {mark}")
            print()

    # Summary table sorted by alias rate.
    pair_results.sort(key=lambda r: (-r[4], r[0], r[2]))
    print("=== Scheme A summary (alias rate per pair) ===\n")
    print(f"  {'store':<14} {'→':<3} {'load':<14}  alias_rate")
    print(f"  {'-'*14}    {'-'*14}  {'-'*10}")
    for (s_l, s_r, l_l, l_r, nm, nt) in pair_results:
        kind = "(same)" if s_r == l_r else "      "
        print(f"  L{s_l:>3} {s_r:<6}{kind} L{l_l:>3} {l_r:<6}    {nm}/{nt}")
    print()

    confirmed_full = [r for r in pair_results if r[4] == r[5]]
    confirmed_partial = [r for r in pair_results if 0 < r[4] < r[5]]

    if confirmed_full:
        print(f"  ✓ {len(confirmed_full)} pair(s) ALIAS in 100% of inputs (definitive v4 candidate):")
        for (s_l, s_r, l_l, l_r, nm, nt) in confirmed_full:
            print(f"    store @L{s_l} on {s_r}  →  load @L{l_l} on {l_r}")
    elif confirmed_partial:
        print(f"  ◐ {len(confirmed_partial)} pair(s) alias on SOME inputs (input-dependent v4):")
        for (s_l, s_r, l_l, l_r, nm, nt) in confirmed_partial[:5]:
            print(f"    store @L{s_l} on {s_r}  →  load @L{l_l} on {l_r}: {nm}/{nt}")
        print("    These could be the v4 trigger — the original violation may")
        print("    have been detected because one specific input hit a collision.")
    else:
        print("  ✗ No pair aliases in any input.")
        print("    The reported violation is likely NOT classical v4 store-bypass,")
        print("    or v4 fires on inputs not in this 'num_inputs' sample. Try")
        print("    raising --num-inputs.")


# ----------------------------------------------------------------------------
# 3. Pair analysis.
# ----------------------------------------------------------------------------
def _analyze_pairs(
    mem_ops: List[Tuple[int, str, str, str]],
    snapshots: List[List[int]],
) -> None:
    """For every store→load pair in program order, report whether their
    base registers happen to share an end-state value."""
    stores = [(ln, base, raw) for (ln, kind, base, raw) in mem_ops if kind == "store"]
    loads = [(ln, base, raw) for (ln, kind, base, raw) in mem_ops if kind == "load"]

    name_to_idx = {n: i for i, n in enumerate(SNAPSHOT_REGS)}

    print(f"\nFound {len(stores)} store(s) and {len(loads)} load(s).")
    print("Considering only store→load pairs where the store precedes the load.\n")

    candidates = []
    for (s_line, s_base, s_raw) in stores:
        for (l_line, l_base, l_raw) in loads:
            if l_line <= s_line:
                continue
            si = name_to_idx[s_base]
            li = name_to_idx[l_base]

            # Across each input, do the two regs hold the same end-state value?
            matches_per_input = []
            for snap in snapshots:
                matches_per_input.append(snap[si] == snap[li])
            n_match = sum(matches_per_input)

            # Also surface the equality-encoding view: are both slots flagged?
            enc_flags = []
            for snap in snapshots:
                enc = _equality_encoding(snap)
                # Both slots flagged AND they share the same value.
                enc_flags.append(enc[si] == 1 and enc[li] == 1 and snap[si] == snap[li])

            candidates.append({
                "store_line": s_line, "store": s_raw, "store_reg": s_base,
                "load_line": l_line, "load": l_raw, "load_reg": l_base,
                "n_match": n_match, "n_inputs": len(snapshots),
                "enc_match": sum(enc_flags),
            })

    # Split: cross-register pairs (interesting for v4) vs same-register
    # pairs (trivially match if reg unchanged between them, no v4 evidence).
    cross = [c for c in candidates if c["store_reg"] != c["load_reg"]]
    same = [c for c in candidates if c["store_reg"] == c["load_reg"]]
    cross.sort(key=lambda c: (-c["n_match"], c["store_line"], c["load_line"]))
    same.sort(key=lambda c: (-c["n_match"], c["store_line"], c["load_line"]))

    cross_matched = [c for c in cross if c["n_match"] > 0]

    print(f"=== Cross-register store→load pairs (potential v4 alias) ===\n")
    if not cross_matched:
        print("  None. No store→load pair with DIFFERENT base registers shows")
        print("  end-state value equality across any input.")
        print()
        print("  → Under scheme C this means 'no detectable alias'. v4 may STILL")
        print("    have fired mid-program if the bypass-load's base register was")
        print("    overwritten before program exit. Confirm with scheme A.")
    else:
        for c in cross_matched:
            bar = "✓" * c["n_match"] + "·" * (c["n_inputs"] - c["n_match"])
            print(f"  store @L{c['store_line']:>3}  base={c['store_reg']}   {c['store']}")
            print(f"  load  @L{c['load_line']:>3}  base={c['load_reg']}   {c['load']}")
            print(f"   end-state {c['store_reg']}=={c['load_reg']}: "
                  f"{c['n_match']}/{c['n_inputs']} inputs  [{bar}]")
            print()

    print(f"\n=== Same-register store→load pairs (trivial match) ===\n")
    print(f"  {len(same)} pair(s) where store and load use the SAME base register.")
    print(f"  These trivially share end-state values if no instruction in between")
    print(f"  overwrote the register; they do NOT prove address aliasing at the")
    print(f"  moment of execution. Listing for reference only:")
    print()
    for c in same[:10]:  # cap at 10 for readability
        print(f"    L{c['store_line']:>3} → L{c['load_line']:>3}  on {c['store_reg']}")
    if len(same) > 10:
        print(f"    ... ({len(same) - 10} more)")


# ----------------------------------------------------------------------------
# 4. Pretty-print snapshots.
# ----------------------------------------------------------------------------
def _print_snapshots(snapshots: List[List[int]]) -> None:
    print("\n=== End-state register snapshots (one row per input) ===\n")
    header = "input |  " + "  ".join(f"{n:>14}" for n in SNAPSHOT_REGS) + "  | enc"
    print(header)
    print("-" * len(header))
    for i, snap in enumerate(snapshots):
        enc = _equality_encoding(snap)
        cells = "  ".join(f"0x{v:012x}" for v in snap)
        enc_str = "".join(str(b) for b in enc)
        print(f"  {i:>3} |  {cells}  | {enc_str}")
    print()


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("asm_path", help="path to a violation .asm file")
    ap.add_argument("--num-inputs", type=int, default=10,
                    help="how many inputs to feed through arch_executor")
    ap.add_argument("--no-scheme-a", action="store_true",
                    help="skip the per-pair scheme-A diagnostic (cheap mode)")
    ap.add_argument("--no-scheme-c", action="store_true",
                    help="skip the end-state scheme-C overview")
    args = ap.parse_args()

    if not os.path.isfile(args.asm_path):
        sys.exit(f"file not found: {args.asm_path}")

    print(f"=== Diagnosing {args.asm_path} ===")

    mem_ops = _parse_violation_asm(args.asm_path)
    if not mem_ops:
        sys.exit("no agent-emitted memory instructions detected — asm filtering may be off.")

    print(f"\nDetected memory instructions ({len(mem_ops)}):")
    for line_no, kind, base, raw in mem_ops:
        print(f"  L{line_no:>3} {kind:>5} on {base}   {raw}")

    if not args.no_scheme_c:
        snapshots = _capture_register_state(args.asm_path, args.num_inputs)
        _print_snapshots(snapshots)
        _analyze_pairs(mem_ops, snapshots)

    if not args.no_scheme_a:
        _run_scheme_a_for_all_cross_pairs(args.asm_path, mem_ops, args.num_inputs)


if __name__ == "__main__":
    main()
