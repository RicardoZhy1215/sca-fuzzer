# Vocabulary
OPERAND_SPACE = ["mov", "add", "mul", "xor", "cmp", "sbb"]
DST_REGS_SPACE = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi"]
SRC_REGS_SPACE = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi"]
IMMS_SPACE = ["0", "1", "2", "3", "4", "5", "6", "7"] #todo: need to extend the IMM Space based on the Sandbox area.

# Index 0 = empty/N/A for regs and imm
EMPTY_REG_ID = 0
EMPTY_IMM_ID = 0

OPCODE_OPERAND_SPEC = {
    "mov": {"dst_reg": "required", "src_reg": "optional", "imm": "optional"},
    "add": {"dst_reg": "required", "src_reg": "optional", "imm": "optional"},
    "mul": {"dst_reg": "forbidden", "src_reg": "required", "imm": "forbidden"},
    "xor": {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmp": {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "sbb": {"dst_reg": "required", "src_reg": "optional", "imm": "optional"},
}
# Constraint: for mov/add/sbb with optional src_reg and optional imm, at least one of src_reg or imm must be non-empty
OPCODE_NEEDS_SRC_OR_IMM = ["mov", "add", "sbb"]
# Exclude dst_reg == src_reg for these opcodes (redundant: mov rax,rax no-op, cmp rax,rax pointless).
# xor rax,rax is kept - it's a useful zeroing idiom.
OPCODE_EXCLUDE_DST_EQ_SRC = ["mov", "add", "cmp", "sbb"]


def _is_reg_empty(reg_id: int) -> bool:
    return reg_id == EMPTY_REG_ID


def _is_imm_empty(imm_id: int) -> bool:
    return imm_id == EMPTY_IMM_ID


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
    # mov/add/sbb: src is EITHER reg OR imm, not both (mutually exclusive)
    if opcode_name in OPCODE_NEEDS_SRC_OR_IMM and (src_req == "optional" and imm_req == "optional"):
        if _is_reg_empty(reg_src_id) and _is_imm_empty(imm_id):
            return False
        if not _is_reg_empty(reg_src_id) and not _is_imm_empty(imm_id):
            return False
    # Exclude dst_reg == src_reg for opcodes where it's redundant (mov/cmp) or pointless
    if opcode_name in OPCODE_EXCLUDE_DST_EQ_SRC:
        if not _is_reg_empty(reg_src_id) and not _is_reg_empty(reg_dst_id) and reg_src_id == reg_dst_id:
            return False
    return True


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


# def get_opcode_mask() -> list:
#     """Mask for opcode step: all opcodes + end_game are allowed."""
#     return [True] * (len(OPERAND_SPACE) + 1)


# def get_reg_src_mask(opcode_id: int) -> list:
#     """Mask for reg_src given opcode_id. Length = n_regs + 1 (empty)."""
#     n_reg = len(SRC_REGS_SPACE)
#     if opcode_id == 0:
#         return [True] + [False] * n_reg
#     opcode_name = OPERAND_SPACE[opcode_id - 1]
#     spec = OPCODE_OPERAND_SPEC.get(opcode_name, {})
#     src_req = spec.get("src_reg", "optional")
#     if src_req == "required":
#         return [False] + [True] * n_reg
#     if src_req == "forbidden":
#         return [True] + [False] * n_reg
#     return [True] * (n_reg + 1)


# def get_reg_dst_mask(opcode_id: int) -> list:
#     """Mask for reg_dst given opcode_id."""
#     n_reg = len(DST_REGS_SPACE)
#     if opcode_id == 0:
#         return [True] + [False] * n_reg
#     opcode_name = OPERAND_SPACE[opcode_id - 1]
#     spec = OPCODE_OPERAND_SPEC.get(opcode_name, {})
#     dst_req = spec.get("dst_reg", "optional")
#     if dst_req == "required":
#         return [False] + [True] * n_reg
#     if dst_req == "forbidden":
#         return [True] + [False] * n_reg
#     return [True] * (n_reg + 1)


# def get_imm_mask(opcode_id: int) -> list:
#     """Mask for imm given opcode_id. Length = n_imm + 1 (empty)."""
#     n_imm = len(IMMS_SPACE)
#     if opcode_id == 0:
#         return [True] + [False] * n_imm
#     opcode_name = OPERAND_SPACE[opcode_id - 1]
#     spec = OPCODE_OPERAND_SPEC.get(opcode_name, {})
#     imm_req = spec.get("imm", "optional")
#     if imm_req == "required":
#         return [False] + [True] * n_imm
#     if imm_req == "forbidden":
#         return [True] + [False] * n_imm
#     return [True] * (n_imm + 1)


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
    from rvzr.tc_components.instruction import Instruction, RegisterOp, ImmediateOp, MemoryOp

    if opcode_id == 0:
        return None
    opcode = OPERAND_SPACE[opcode_id - 1]
    rd = DST_REGS_SPACE[reg_dst_id - 1] if reg_dst_id else None
    rs = SRC_REGS_SPACE[reg_src_id - 1] if reg_src_id else None
    imm_val = IMMS_SPACE[imm_id - 1] if imm_id else None

    if opcode == "mov":
        if imm_val is not None:
            return Instruction("mov", False, "", False).add_op(
                MemoryOp(rd, 64, False, True)
            ).add_op(ImmediateOp(imm_val, 64))
        return Instruction("mov", False, "", False).add_op(
            RegisterOp(rd, 64, False, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "add":
        if imm_val is not None:
            return Instruction("add", False, "", False).add_op(
                RegisterOp(rd, 64, False, True)
            ).add_op(ImmediateOp(imm_val, 64))
        return Instruction("add", False, "", False).add_op(
            RegisterOp(rd, 64, False, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "mul":
        return Instruction("mul", False, "", False).add_op(
            MemoryOp(rs, 32, False, True)
        )

    if opcode == "xor":
        return Instruction("xor", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "cmp":
        return Instruction("cmp", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "sbb":
        if imm_val is not None:
            return Instruction("sbb", False, "", False).add_op(
                MemoryOp(rd, 64, True, True)
            ).add_op(ImmediateOp(imm_val, 8))
        return Instruction("sbb", False, "", False).add_op(
            MemoryOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    return None


# Backward compatibility aliases
Operand_Space = OPERAND_SPACE
Dst_Regs_Space = DST_REGS_SPACE
Src_Regs_Space = SRC_REGS_SPACE
Imms_Space = IMMS_SPACE
