"""
Detailed statistics for inst_space action space filtering.
Run from hierarchical_testing: python inst_space_stats.py
"""
import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from inst_space import (
    OPERAND_SPACE,
    DST_REGS_SPACE,
    IMMS_SPACE,
    OPCODE_OPERAND_SPEC,
    OPCODE_NEEDS_SRC_OR_IMM,
    OPCODE_EXCLUDE_DST_EQ_SRC,
    compute_action_space_stats,
    build_action_to_tuple,
)


def main():
    stats = compute_action_space_stats()
    action_to_tuple = build_action_to_tuple()

    n_reg = len(DST_REGS_SPACE)
    n_imm = len(IMMS_SPACE)

    print("=" * 70)
    print("inst_space Action Space Statistics")
    print("=" * 70)
    print()
    print("## Vocabulary")
    print("-" * 40)
    print(f"  Opcodes:        {OPERAND_SPACE}")
    print(f"  Registers:      {DST_REGS_SPACE} (+ empty)")
    print(f"  Immediates:     {IMMS_SPACE} (+ empty)")
    print(f"  opcode_id:      1..{len(OPERAND_SPACE)} (instruction), 0 (end_game)")
    print(f"  reg_src/dst:    0..{n_reg} (0=empty, 1..{n_reg}=reg)")
    print(f"  imm:            0..{n_imm} (0=empty, 1..{n_imm}=imm)")
    print()

    total_raw = stats["total_raw"]
    print("## Total Raw Combinations (before any rules)")
    print("-" * 40)
    print(f"  Formula:  {len(OPERAND_SPACE)} opcodes × {n_reg + 1} reg_src × {n_reg + 1} reg_dst × {n_imm + 1} imm")
    print(f"  Total:    {total_raw}")
    print()

    print("## Rules and Exclusions")
    print("-" * 40)
    print()
    print("  Rule 1: OPCODE_OPERAND_SPEC (required/optional/forbidden)")
    print("    - dst_reg, src_reg, imm must satisfy per-opcode spec")
    for op, spec in OPCODE_OPERAND_SPEC.items():
        print(f"      {op}: dst={spec['dst_reg']}, src={spec['src_reg']}, imm={spec['imm']}")
    excl1 = stats["excluded_by_rule_operand_spec"]
    print(f"    Excluded: {excl1}")
    print(f"    Remaining: {total_raw - excl1}")
    print()

    print("  Rule 2: OPCODE_NEEDS_SRC_OR_IMM")
    print(f"    - For {OPCODE_NEEDS_SRC_OR_IMM}: src_reg and imm are mutually exclusive")
    print("    - At least one of src_reg or imm must be non-empty")
    print("    - Not both non-empty")
    excl2 = stats["excluded_by_rule_src_or_imm"]
    after_r1 = total_raw - excl1
    print(f"    Excluded (among those passing Rule 1): {excl2}")
    print(f"    Remaining: {after_r1 - excl2}")
    print()

    print("  Rule 3: OPCODE_EXCLUDE_DST_EQ_SRC")
    print(f"    - For {OPCODE_EXCLUDE_DST_EQ_SRC}: exclude dst_reg == src_reg when both non-empty")
    excl3 = stats["excluded_by_rule_exclude_dst_eq_src"]
    after_r2 = after_r1 - excl2
    print(f"    Excluded (among those passing Rule 1&2): {excl3}")
    print(f"    Remaining: {after_r2 - excl3}")
    print()

    print("## Summary")
    print("-" * 40)
    print(f"  Total raw combinations:           {total_raw}")
    print(f"  Excluded by Rule 1 (operand):     {excl1}")
    print(f"  Excluded by Rule 2 (src/imm):     {excl2}")
    print(f"  Excluded by Rule 3 (dst==src):    {excl3}")
    print(f"  Total excluded:                   {excl1 + excl2 + excl3}")
    print(f"  Remaining (instructions):         {stats['remaining_instructions']}")
    print(f"  Remaining (+ end_game):           {stats['remaining_with_end_game']}")
    print(f"  build_action_to_tuple() length:   {len(action_to_tuple)}")
    print()

    print("## Per-Opcode Breakdown")
    print("-" * 40)
    print(f"  {'Opcode':<8} {'Raw':>8} {'After R1':>10} {'After R2':>10} {'After R3':>10}")
    print("  " + "-" * 50)
    for op in OPERAND_SPACE:
        p = stats["per_opcode"][op]
        print(f"  {op:<8} {p['raw']:>8} {p['after_r1']:>10} {p['after_r2']:>10} {p['after_r3']:>10}")
    print("  " + "-" * 50)
    print(f"  {'Total':<8} {total_raw:>8} {total_raw - excl1:>10} {after_r1 - excl2:>10} {stats['remaining_instructions']:>10}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
