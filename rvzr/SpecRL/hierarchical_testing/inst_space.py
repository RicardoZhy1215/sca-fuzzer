from rvzr.config import CONF

# Vocabulary
# Split mov into explicit operand-mode variants so each opcode has stable semantics.
#
# Stage-1 action space for Spectre v4 curriculum: only keep opcodes that are
# useful for building an SSB gadget (slow-addr producer / store / bypass load /
# transmitter).  The full 22-opcode list is preserved in _FULL_OPERAND_SPACE so
# that it can be restored later as stage-2 fine-tuning:
#
#   _FULL_OPERAND_SPACE = [
#       "mov_rr", "mov_ri", "mov_rm", "mov_mr", "mov_mi",
#       "add_rr", "add_ri", "add_rm", "add_mr", "add_mi",
#       "mul", "div", "xor",
#       "cmp_rr", "cmp_rm", "cmp_mr",
#       "sub_rr", "sub_ri", "sub_rm", "sub_mr", "sub_mi",
#   ]
#
# Kept opcodes (stage-1):
#   mov_rr : glue / register setup
#   mov_ri : stage register with constant (used with care - smashes base chains)
#   mov_rm : load (also acts as slow-addr producer when followed by store)
#   mov_mr : store (candidate victim of SSB)
#   mov_mi : store with immediate value (also a candidate victim)
#   add_rm : load-ALU (slow-addr producer)
#   add_mr : RMW store (alternative store form)
OPERAND_SPACE = [
    "mov_rr",  # mov reg, reg
    "mov_ri",  # mov reg, imm
    "mov_rm",  # mov reg, [reg]    -- bypass load / slow-addr producer
    "mov_mr",  # mov [reg], reg    -- store
    "mov_mi",  # mov [reg], imm    -- store
    "add_rm",  # add reg, [reg]    -- slow-addr producer
    "add_mr",  # add [reg], reg    -- RMW store
    "lea_rrr", # lea rd, [rs + rd + 1]  -- serial slow-addr chain (v4 producer)
    "mul",     # mul qword ptr [rs]     -- long-latency producer (rax:rdx)
    "xor_rr",  # xor rd, rs             -- fast reg reset / zeroing idiom
]
DST_REGS_SPACE = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi"] # add more regs like al, bl,eax, etc.
SRC_REGS_SPACE = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi"]
_IMM_SMALL_LITERALS = range(8)
_SANDBOX_IMM_STEP = 8
_SANDBOX_IMM_LIMIT = CONF.input_main_region_size + CONF.input_faulty_region_size
IMMS_SPACE = [str(imm) for imm in _IMM_SMALL_LITERALS]
IMMS_SPACE.extend(str(imm) for imm in range(_SANDBOX_IMM_STEP, _SANDBOX_IMM_LIMIT, _SANDBOX_IMM_STEP))

# Index 0 = empty/N/A for regs and imm
EMPTY_REG_ID = 0
EMPTY_IMM_ID = 0

OPCODE_OPERAND_SPEC = {
    "mov_rr":  {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "mov_ri":  {"dst_reg": "required", "src_reg": "forbidden", "imm": "required"},
    "mov_rm":  {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "mov_mr":  {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "mov_mi":  {"dst_reg": "required", "src_reg": "forbidden", "imm": "required"},
    "add_rm":  {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "add_mr":  {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    # lea rd, [rs + rd + 1]  -> dst=required, src=required, imm forbidden
    "lea_rrr": {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    # mul qword ptr [rs]  -> implicit rax:rdx dst, only src_reg as mem base
    "mul":     {"dst_reg": "forbidden", "src_reg": "required", "imm": "forbidden"},
    # xor rd, rs  (dst==src zeroes; explicitly allowed for the zeroing idiom)
    "xor_rr":  {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
}
# Exclude dst_reg == src_reg for opcodes where it is pointless (mov rax,rax no-op).
OPCODE_EXCLUDE_DST_EQ_SRC = ["mov_rr"]


def _is_reg_empty(reg_id: int) -> bool:
    return reg_id == EMPTY_REG_ID


def _is_imm_empty(imm_id: int) -> bool:
    return imm_id == EMPTY_IMM_ID


def _check_rule_operand_spec(opcode_id: int, reg_src_id: int, reg_dst_id: int, imm_id: int) -> bool:
    """Rule 1: OPCODE_OPERAND_SPEC (required/optional/forbidden)."""
    if opcode_id < 1 or opcode_id > len(OPERAND_SPACE):
        return False
    opcode_name = OPERAND_SPACE[opcode_id - 1]
    spec = OPCODE_OPERAND_SPEC.get(opcode_name, {})
    dst_req = spec.get("dst_reg", "optional")
    src_req = spec.get("src_reg", "optional")
    imm_req = spec.get("imm", "optional")

    def check(field_req: str, is_empty: bool) -> bool:
        if field_req == "required":
            return not is_empty
        if field_req == "forbidden":
            return is_empty
        return True

    if not check(dst_req, _is_reg_empty(reg_dst_id)):
        return False
    if not check(src_req, _is_reg_empty(reg_src_id)):
        return False
    if not check(imm_req, _is_imm_empty(imm_id)):
        return False
    return True


def _check_rule_src_or_imm(opcode_id: int, reg_src_id: int, reg_dst_id: int, imm_id: int) -> bool:
    """Rule 2: reserved for opcodes with coupled src/imm semantics."""
    return True


def _check_rule_exclude_dst_eq_src(opcode_id: int, reg_src_id: int, reg_dst_id: int, imm_id: int) -> bool:
    """Rule 3: OPCODE_EXCLUDE_DST_EQ_SRC - exclude dst==src for mov/add/cmp/sub."""
    if opcode_id < 1 or opcode_id > len(OPERAND_SPACE):
        return True
    opcode_name = OPERAND_SPACE[opcode_id - 1]
    if opcode_name not in OPCODE_EXCLUDE_DST_EQ_SRC:
        return True
    if not _is_reg_empty(reg_src_id) and not _is_reg_empty(reg_dst_id) and reg_src_id == reg_dst_id:
        return False
    return True


def is_action_legal(opcode_id: int, reg_src_id: int, reg_dst_id: int, imm_id: int) -> bool:
    """
    Check if (opcode_id, reg_src_id, reg_dst_id, imm_id) is legal per OPCODE_OPERAND_SPEC.
    opcode_id: 0=end_game, 1..len(OPERAND_SPACE)=opcode index
    reg_src_id, reg_dst_id: 0=empty, 1..len(REG_SPACE)=reg index
    imm_id: 0=empty, 1..len(IMMS_SPACE)=imm index
    """
    if opcode_id == 0:
        return reg_src_id == 0 and reg_dst_id == 0 and imm_id == 0
    if opcode_id < 1 or opcode_id > len(OPERAND_SPACE):
        return False
    opcode_name = OPERAND_SPACE[opcode_id - 1]
    spec = OPCODE_OPERAND_SPEC.get(opcode_name, {})
    dst_req = spec.get("dst_reg", "optional")
    src_req = spec.get("src_reg", "optional")
    imm_req = spec.get("imm", "optional")

    def check(field_req: str, is_empty: bool) -> bool:
        if field_req == "required":
            return not is_empty
        if field_req == "forbidden":
            return is_empty
        return True

    if not check(dst_req, _is_reg_empty(reg_dst_id)):
        return False
    if not check(src_req, _is_reg_empty(reg_src_id)):
        return False
    if not check(imm_req, _is_imm_empty(imm_id)):
        return False
    # Exclude dst_reg == src_reg for opcodes where it's redundant (mov_rr/cmp_rr) or pointless
    if opcode_name in OPCODE_EXCLUDE_DST_EQ_SRC:
        if not _is_reg_empty(reg_src_id) and not _is_reg_empty(reg_dst_id) and reg_src_id == reg_dst_id:
            return False
    return True


def compute_action_space_stats() -> dict:
    """
    Compute detailed statistics on action space filtering.
    Returns dict with: total_raw, excluded_by_rule_*, remaining, per_opcode.
    """
    n_op = len(OPERAND_SPACE)
    n_reg = len(DST_REGS_SPACE)
    n_imm = len(IMMS_SPACE)
    # Raw: opcode 1..len(OPERAND_SPACE), reg_src 0..n_reg, reg_dst 0..n_reg, imm 0..n_imm
    total_raw = n_op * (n_reg + 1) * (n_reg + 1) * (n_imm + 1)

    pass_r1 = 0
    pass_r1_r2 = 0
    pass_all = 0
    excluded_by_r1 = 0
    excluded_by_r2 = 0
    excluded_by_r3 = 0
    per_opcode = {op: {"raw": 0, "after_r1": 0, "after_r2": 0, "after_r3": 0} for op in OPERAND_SPACE}

    for opcode_id in range(1, n_op + 1):
        opcode_name = OPERAND_SPACE[opcode_id - 1]
        for rd in range(n_reg + 1):
            for rs in range(n_reg + 1):
                for imm in range(n_imm + 1):
                    per_opcode[opcode_name]["raw"] += 1
                    if not _check_rule_operand_spec(opcode_id, rs, rd, imm):
                        excluded_by_r1 += 1
                        continue
                    pass_r1 += 1
                    per_opcode[opcode_name]["after_r1"] += 1
                    if not _check_rule_src_or_imm(opcode_id, rs, rd, imm):
                        excluded_by_r2 += 1
                        continue
                    pass_r1_r2 += 1
                    per_opcode[opcode_name]["after_r2"] += 1
                    if not _check_rule_exclude_dst_eq_src(opcode_id, rs, rd, imm):
                        excluded_by_r3 += 1
                        continue
                    pass_all += 1
                    per_opcode[opcode_name]["after_r3"] += 1

    return {
        "total_raw": total_raw,
        "excluded_by_rule_operand_spec": excluded_by_r1,
        "excluded_by_rule_src_or_imm": excluded_by_r2,
        "excluded_by_rule_exclude_dst_eq_src": excluded_by_r3,
        "remaining_instructions": pass_all,
        "remaining_with_end_game": pass_all + 1,
        "per_opcode": per_opcode,
    }


def build_action_to_tuple() -> list:
    """
    Enumerate all legal (opcode_id, reg_src_id, reg_dst_id, imm_id) and return action_to_tuple.
    Last action is end_game (0,0,0,0) to match env convention.
    """
    n_op = len(OPERAND_SPACE)
    n_reg = len(DST_REGS_SPACE)
    n_imm = len(IMMS_SPACE)
    result = []
    for opcode_id in range(1, n_op + 1):
        for rd in range(n_reg + 1):
            for rs in range(n_reg + 1):
                for imm in range(n_imm + 1):
                    if is_action_legal(opcode_id, rs, rd, imm):
                        result.append((opcode_id, rs, rd, imm))
    result.append((0, 0, 0, 0))
    return result


def get_num_opcodes() -> int:
    return len(OPERAND_SPACE) + 1


def get_num_regs() -> int:
    return len(DST_REGS_SPACE) + 1


def get_num_imms() -> int:
    return len(IMMS_SPACE) + 1


def compute_flat_action_mask(action_to_tuple: list) -> list:
    """
    For each action in action_to_tuple, check if it is legal.
    Returns a boolean list of same length. Used to mask logits.
    """
    return [is_action_legal(o, rs, rd, imm) for o, rs, rd, imm in action_to_tuple]


def get_opcode_mask() -> list:
    """Mask for opcode step: all opcodes + end_game are allowed."""
    return [True] * (len(OPERAND_SPACE) + 1)


def get_reg_src_mask(opcode_id: int) -> list:
    """Mask for reg_src given opcode_id. Length = n_regs + 1 (index 0 = empty)."""
    n_reg = len(SRC_REGS_SPACE)
    if opcode_id == 0:
        return [True] + [False] * n_reg
    opcode_name = OPERAND_SPACE[opcode_id - 1]
    spec = OPCODE_OPERAND_SPEC.get(opcode_name, {})
    src_req = spec.get("src_reg", "optional")
    if src_req == "required":
        return [False] + [True] * n_reg
    if src_req == "forbidden":
        return [True] + [False] * n_reg
    return [True] * (n_reg + 1)


def get_reg_dst_mask(opcode_id: int, reg_src_id: int = 0) -> list:
    """Mask for reg_dst given opcode_id and reg_src_id (for OPCODE_EXCLUDE_DST_EQ_SRC)."""
    n_reg = len(DST_REGS_SPACE)
    if opcode_id == 0:
        return [True] + [False] * n_reg
    opcode_name = OPERAND_SPACE[opcode_id - 1]
    spec = OPCODE_OPERAND_SPEC.get(opcode_name, {})
    dst_req = spec.get("dst_reg", "optional")
    base = [True] * (n_reg + 1)
    if dst_req == "required":
        base = [False] + [True] * n_reg
    elif dst_req == "forbidden":
        base = [True] + [False] * n_reg
    # Exclude dst==src for mov_rr/add/cmp/sub
    if opcode_name in OPCODE_EXCLUDE_DST_EQ_SRC and reg_src_id > 0:
        if reg_src_id <= n_reg and base[reg_src_id]:
            base = base[:]
            base[reg_src_id] = False
    return base


def get_imm_mask(opcode_id: int, reg_src_id: int = 0) -> list:
    """
    Mask for imm given opcode_id and reg_src_id.
    """
    n_imm = len(IMMS_SPACE)
    if opcode_id == 0:
        return [True] + [False] * n_imm
    opcode_name = OPERAND_SPACE[opcode_id - 1]
    spec = OPCODE_OPERAND_SPEC.get(opcode_name, {})
    imm_req = spec.get("imm", "optional")
    if imm_req == "required":
        return [False] + [True] * n_imm
    if imm_req == "forbidden":
        return [True] + [False] * n_imm
    return [True] * (n_imm + 1)


def build_action_to_tuple_masked() -> tuple:
    """
    Build action_to_tuple with only legal actions (action masking applied at enumeration).
    Returns (action_to_tuple, flat_action_mask) where mask is all True since we only enumerate legal.
    """
    action_to_tuple = build_action_to_tuple()
    mask = compute_flat_action_mask(action_to_tuple)
    return action_to_tuple, mask


def tuple_to_instruction(opcode_id: int, reg_src_id: int, reg_dst_id: int, imm_id: int):
    """
    Convert (opcode_id, reg_src_id, reg_dst_id, imm_id) to rvzr Instruction.
    Returns None for end_game (0,0,0,0).
    """
    from rvzr.tc_components.instruction import Instruction, RegisterOp, ImmediateOp, MemoryOp, AgenOp

    if opcode_id == 0:
        return None
    opcode = OPERAND_SPACE[opcode_id - 1]
    rd = DST_REGS_SPACE[reg_dst_id - 1] if reg_dst_id else None
    rs = SRC_REGS_SPACE[reg_src_id - 1] if reg_src_id else None
    imm_val = IMMS_SPACE[imm_id - 1] if imm_id else None

    if opcode == "mov_rr":
        return Instruction("mov", False, "", False).add_op(
            RegisterOp(rd, 64, False, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "mov_ri":
        return Instruction("mov", False, "", False).add_op(
            RegisterOp(rd, 64, False, True)
        ).add_op(ImmediateOp(imm_val, 64))

    if opcode == "mov_rm":
        return Instruction("mov", False, "", False).add_op(
            RegisterOp(rd, 64, False, True)
        ).add_op(MemoryOp(rs, 64, True, False))

    if opcode == "mov_mr":
        return Instruction("mov", False, "", False).add_op(
            MemoryOp(rd, 64, False, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "mov_mi":
        return Instruction("mov", False, "", False).add_op(
            MemoryOp(rd, 64, False, True)
        ).add_op(ImmediateOp(imm_val, 64))

    if opcode == "add_rr":
        return Instruction("add", False, "", False).add_op(
            RegisterOp(rd, 64, False, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "add_ri":
        return Instruction("add", False, "", False).add_op(
            RegisterOp(rd, 64, False, True)
        ).add_op(ImmediateOp(imm_val, 64))

    if opcode == "add_rm":
        return Instruction("add", False, "", False).add_op(
            RegisterOp(rd, 64, False, True)
        ).add_op(MemoryOp(rs, 64, True, False))

    if opcode == "add_mr":
        return Instruction("add", False, "", False).add_op(
            MemoryOp(rd, 64, False, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "add_mi":
        return Instruction("add", False, "", False).add_op(
            MemoryOp(rd, 64, False, True)
        ).add_op(ImmediateOp(imm_val, 64))

    if opcode == "mul":
        # mul qword ptr [rs] -> writes rax:rdx implicitly. Using 64-bit width so
        # the full 64x64->128 result surfaces and the op actually costs latency.
        return Instruction("mul", False, "", False).add_op(
            MemoryOp(rs, 64, False, True)
        )

    if opcode == "div":
        return Instruction("div", False, "", False).add_op(
            MemoryOp(rs, 8, False, True)
        )

    if opcode == "xor_rr":
        return Instruction("xor", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "lea_rrr":
        # Slow-addr chain: rd = rs + rd + 1. rd appears on both sides so the
        # dependency serializes across multiple lea_rrr's, matching the
        # canonical Spectre-v4 gadget (see tests/x86_tests/asm/spectre_v4.asm).
        # Using AgenOp so sandbox pass leaves the address expression alone.
        return Instruction("lea", False, "", False).add_op(
            RegisterOp(rd, 64, False, True)
        ).add_op(AgenOp(f"{rs} + {rd} + 1", 64))

    # if opcode == "cmp":
    #     return Instruction("cmp", False, "", False).add_op(
    #         MemoryOp(rd, 32, True, True)
    #     ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "cmp_rr":
        return Instruction("cmp", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "cmp_rm":
        return Instruction("cmp", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(MemoryOp(rs, 64, True, False))

    if opcode == "cmp_mr":
        return Instruction("cmp", False, "", False).add_op(
            MemoryOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "sub_rr":
        return Instruction("sub", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "sub_ri":
        return Instruction("sub", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(ImmediateOp(imm_val, 8))

    if opcode == "sub_rm":
        return Instruction("sub", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(MemoryOp(rs, 64, True, False))

    if opcode == "sub_mr":
        return Instruction("sub", False, "", False).add_op(
            MemoryOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "sub_mi":
        return Instruction("sub", False, "", False).add_op(
            MemoryOp(rd, 64, True, True)
        ).add_op(ImmediateOp(imm_val, 8))

    return None


# Backward compatibility aliases
Operand_Space = OPERAND_SPACE
Dst_Regs_Space = DST_REGS_SPACE
Src_Regs_Space = SRC_REGS_SPACE
Imms_Space = IMMS_SPACE
