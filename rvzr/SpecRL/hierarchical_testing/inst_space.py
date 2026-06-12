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
    # ---- new pure-GPR opcodes (no XMM/MMX, no new register classes) ----
    # BASE-BINARY div/idiv — ZDI/DSS probe; long latency + microcoded.
    "div_r",            # div  reg64               -- implicit rax:rdx / rax
    # "idiv_r",           # idiv reg64               -- signed counterpart
    # BASE-SEMAPHORE atomics — store-forwarding / MDS class probes.
    "lock_xadd_mr",     # lock xadd [reg64], reg64
    "lock_xchg_mr",     # xchg [reg64], reg64      -- implicit lock when mem operand
    "lock_cmpxchg_mr",  # lock cmpxchg [reg64], reg64  -- implicit rax
    "lock_add_mr",      # lock add  [reg64], reg64
    # BASE-STRINGOP — rep-prefixed loops; SCO and MDS-frontend probes.
    # All four use implicit registers (RSI/RDI/RCX/AL), so the action's
    # reg_src/reg_dst/imm slots are all forced to empty.
    # "rep_movsb",        # repe movsb  -- DS:[RSI] -> ES:[RDI], iterate RCX
    # "rep_stosb",        # repe stosb  -- AL -> ES:[RDI], iterate RCX
    # "rep_cmpsb",        # repe cmpsb  -- DS:[RSI] cmp ES:[RDI], iterate RCX
    # "rep_scasb",        # repe scasb  -- AL cmp ES:[RDI], iterate RCX
    # BASE-CMOV — conditional moves; core transient-compute primitive.
    # The _rm form (cmov reg64, [reg64]) puts the load on the dependent
    # branch, so the cache footprint is gated on the flag-condition outcome
    # — exactly the shape side-channel attacks exploit.
    # cmova is a mnemonic alias for cmovnbe (same opcode 0F 47); base.json
    # only carries the cmovnbe spelling, so the underlying name we emit is
    # cmovnbe even though the action label reads "cmova_rm".
    "cmovz_rm",         # cmov reg64, [reg64]   if ZF=1
    "cmovnz_rm",        # cmov reg64, [reg64]   if ZF=0
    "cmovb_rm",         # cmov reg64, [reg64]   if CF=1   (unsigned below)
    "cmova_rm",         # cmov reg64, [reg64]   if CF=0 and ZF=0  (alias of cmovnbe)
    # BASE-BINARY — fill out the ALU. sub_rr/sbb_rr complete the add/sub
    # symmetry and give the agent CF-consuming ops (sbb reads carry).
    "sub_rr",           # sub reg64, reg64
    "sbb_rr",           # sbb reg64, reg64   -- reads CF (post-cmp/add/sub state)
    # ---- NEW GPR opcodes (all base-ISA, Unicorn-modelable, taint-clean) ----
    # BASE-BINARY / BASE-LOGICAL — flag producers + ALU completers.
    "cmp_rr",           # cmp reg64, reg64       -- sets flags, no write (feeds cmov/setcc)
    "cmp_rm",           # cmp reg64, [reg64]     -- compare against memory
    "cmp_mr",           # cmp [reg64], reg64     -- compare memory against reg
    "test_rr",          # test reg64, reg64      -- AND-based flags, no write
    "and_rr",           # and reg64, reg64
    "or_rr",            # or  reg64, reg64
    "adc_rr",           # adc reg64, reg64       -- reads CF (carry chain, like sbb)
    "imul_rr",          # imul reg64, reg64      -- signed 2-op mul (no rdx:rax clobber)
    "not_r",            # not reg64              -- unary
    "neg_r",            # neg reg64              -- unary, sets CF
    "inc_r",            # inc reg64              -- unary (no CF write)
    "dec_r",            # dec reg64              -- unary (no CF write)
    # BASE-CMOV — complete the flag-condition menu (was only z/nz/b/a).
    "cmovbe_rm",        # cmov reg64, [reg64]    if CF=1 or ZF=1
    "cmovl_rm",         # cmov reg64, [reg64]    if SF!=OF (signed <)
    "cmovle_rm",        # cmov reg64, [reg64]    if ZF=1 or SF!=OF (signed <=)
    "cmovnl_rm",        # cmov reg64, [reg64]    if SF=OF (signed >=)
    "cmovnle_rm",       # cmov reg64, [reg64]    if ZF=0 and SF=OF (signed >)
    "cmovs_rm",         # cmov reg64, [reg64]    if SF=1
    "cmovns_rm",        # cmov reg64, [reg64]    if SF=0
    # BASE-SETCC — flag -> byte (partial-reg write); flag-gated data w/o a branch.
    "setz_r",           # setz  reg8   (low byte of a GPR)
    "setnz_r",          # setnz reg8
    "setb_r",           # setb  reg8
    "setl_r",           # setl  reg8
    # BASE-DATAXFER — sub-width loads (zero/sign extend a byte from memory).
    "movzx_rm",         # movzx reg64, byte ptr [reg64]   -- zero-extend byte load
    "movsx_rm",         # movsx reg64, byte ptr [reg64]   -- sign-extend byte load
    # BASE-SEMAPHORE/LOGICAL atomics — more store-forwarding / MDS probes.
    "lock_sub_mr",      # lock sub [reg64], reg64
    "lock_and_mr",      # lock and [reg64], reg64
    "lock_or_mr",       # lock or  [reg64], reg64
    "lock_xor_mr",      # lock xor [reg64], reg64
    "lock_inc_m",       # lock inc [reg64]
    "lock_dec_m",       # lock dec [reg64]
    # BASE-LOGICAL atomic — single-operand RMW; widens the SEMAPHORE menu.
    # "lock_not_m",       # lock not [reg64]
    # ---- XMM (SSE / SSE2) — DISABLED ----
    # Revizor's model-based pipeline can't handle SIMD faithfully:
    #   (1) Unicorn's SSE/YMM emulation != real CPU -> architectural mismatch (bugs/ w/ report)
    #   (2) taint_tracker under-taints xmm regs (high 64 bits) -> "fast path != full" taint bug
    # Both flood bugs/ and XMM can't carry the GPR address-dependent v4 transmitter anyway.
    # To re-enable: uncomment these opcodes, their OPCODE_OPERAND_SPEC entries below,
    # and restore the xmm names in XMM_REGS_SPACE.
    # "movdqu_xm",        # movdqu xmm, [reg64]   -- 128-bit load (XMM <- mem)
    # "movdqu_mx",        # movdqu [reg64], xmm   -- 128-bit store (mem <- XMM)
    # "movups_xm",        # movups xmm, [reg64]   -- 128-bit load (FP path)
    # "movups_mx",        # movups [reg64], xmm   -- 128-bit store (FP path)
    # "movq_xr",          # movq   xmm, reg64     -- GPR -> XMM (low 64)
    # "movd_xr",          # movd   xmm, reg32     -- GPR -> XMM (low 32)
    # "pxor_xx",          # pxor   xmm, xmm       -- XMM XOR
    # "paddq_xx",         # paddq  xmm, xmm       -- 2x packed-uint64 add
    # "pmuludq_xx",       # pmuludq xmm, xmm      -- 2x 32->64 unsigned multiply
]
# Register pools. Layout:
#   indices 1..6  : GPR  (rax, rbx, rcx, rdx, rsi, rdi)
#   indices 7..14 : XMM  (xmm0..xmm7)
#   index 0       : empty / N/A
# OPCODE_OPERAND_SPEC carries a reg-class tag ("required:gpr" / "required:xmm")
# that the mask functions use to restrict each opcode to the right slice.
GPR_REGS_SPACE = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi"]
# XMM disabled (see DISABLED XMM opcodes above). To re-enable, restore:
#   XMM_REGS_SPACE = ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"]
XMM_REGS_SPACE = []
NUM_GPR = len(GPR_REGS_SPACE)
NUM_XMM = len(XMM_REGS_SPACE)
DST_REGS_SPACE = GPR_REGS_SPACE + XMM_REGS_SPACE
SRC_REGS_SPACE = GPR_REGS_SPACE + XMM_REGS_SPACE
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
    # ---- new pure-GPR opcodes ----
    # div/idiv reg64 -> rax:rdx implicit, only src_reg as divisor
    "div_r":           {"dst_reg": "forbidden", "src_reg": "required", "imm": "forbidden"},
    "idiv_r":          {"dst_reg": "forbidden", "src_reg": "required", "imm": "forbidden"},
    # lock xadd/xchg/cmpxchg/add [reg], reg : dst_reg=mem base, src_reg=value
    "lock_xadd_mr":    {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "lock_xchg_mr":    {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "lock_cmpxchg_mr": {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "lock_add_mr":     {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    # rep movsb/stosb/cmpsb/scasb : everything implicit, all slots empty.
    # "rep_movsb":       {"dst_reg": "forbidden", "src_reg": "forbidden", "imm": "forbidden"},
    # "rep_stosb":       {"dst_reg": "forbidden", "src_reg": "forbidden", "imm": "forbidden"},
    # "rep_cmpsb":       {"dst_reg": "forbidden", "src_reg": "forbidden", "imm": "forbidden"},
    # "rep_scasb":       {"dst_reg": "forbidden", "src_reg": "forbidden", "imm": "forbidden"},
    # cmovcc reg64, [reg64] : dst_reg=GPR, src_reg=mem base.
    "cmovz_rm":        {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmovnz_rm":       {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmovb_rm":        {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmova_rm":        {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    # sub/sbb reg64, reg64. dst==src is allowed (sub r,r zeros, sbb r,r yields 0 or -1 based on CF).
    "sub_rr":          {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "sbb_rr":          {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    # ---- NEW GPR opcodes ----
    # cmp/test/and/or/adc/imul reg64, reg64  (or reg,[mem] / [mem],reg for cmp).
    "cmp_rr":          {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmp_rm":          {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmp_mr":          {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "test_rr":         {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "and_rr":          {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "or_rr":           {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "adc_rr":          {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "imul_rr":         {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    # unary reg64 : only dst.
    "not_r":           {"dst_reg": "required", "src_reg": "forbidden", "imm": "forbidden"},
    "neg_r":           {"dst_reg": "required", "src_reg": "forbidden", "imm": "forbidden"},
    "inc_r":           {"dst_reg": "required", "src_reg": "forbidden", "imm": "forbidden"},
    "dec_r":           {"dst_reg": "required", "src_reg": "forbidden", "imm": "forbidden"},
    # cmovcc reg64, [reg64] : dst_reg=GPR, src_reg=mem base.
    "cmovbe_rm":       {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmovl_rm":        {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmovle_rm":       {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmovnl_rm":       {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmovnle_rm":      {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmovs_rm":        {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "cmovns_rm":       {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    # setcc reg8 : only dst (the GPR whose low byte gets the flag).
    "setz_r":          {"dst_reg": "required", "src_reg": "forbidden", "imm": "forbidden"},
    "setnz_r":         {"dst_reg": "required", "src_reg": "forbidden", "imm": "forbidden"},
    "setb_r":          {"dst_reg": "required", "src_reg": "forbidden", "imm": "forbidden"},
    "setl_r":          {"dst_reg": "required", "src_reg": "forbidden", "imm": "forbidden"},
    # movzx/movsx reg64, [reg64] : dst=GPR, src=mem base (byte load).
    "movzx_rm":        {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "movsx_rm":        {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    # lock <op> [reg64], reg64 : dst=mem base, src=value reg.
    "lock_sub_mr":     {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "lock_and_mr":     {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "lock_or_mr":      {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    "lock_xor_mr":     {"dst_reg": "required", "src_reg": "required", "imm": "forbidden"},
    # lock inc/dec [reg64] : single mem operand.
    "lock_inc_m":      {"dst_reg": "required", "src_reg": "forbidden", "imm": "forbidden"},
    "lock_dec_m":      {"dst_reg": "required", "src_reg": "forbidden", "imm": "forbidden"},
    # lock not [reg64] : single mem operand, no value register.
    "lock_not_m":      {"dst_reg": "required", "src_reg": "forbidden", "imm": "forbidden"},
    # ---- XMM opcodes. Reg-class tags ("required:gpr" / "required:xmm") are
    # honored by get_reg_*_mask / is_action_legal so GPR slots only get GPR
    # and XMM slots only get XMM.  "required" without a class is implicitly
    # "required:gpr" (backward compat with existing entries above).
    # ---- XMM specs DISABLED (see DISABLED XMM opcodes in OPERAND_SPACE) ----
    # "movdqu_xm":       {"dst_reg": "required:xmm", "src_reg": "required:gpr", "imm": "forbidden"},
    # "movdqu_mx":       {"dst_reg": "required:gpr", "src_reg": "required:xmm", "imm": "forbidden"},
    # "movups_xm":       {"dst_reg": "required:xmm", "src_reg": "required:gpr", "imm": "forbidden"},
    # "movups_mx":       {"dst_reg": "required:gpr", "src_reg": "required:xmm", "imm": "forbidden"},
    # "movq_xr":         {"dst_reg": "required:xmm", "src_reg": "required:gpr", "imm": "forbidden"},
    # "movd_xr":         {"dst_reg": "required:xmm", "src_reg": "required:gpr", "imm": "forbidden"},
    # "pxor_xx":         {"dst_reg": "required:xmm", "src_reg": "required:xmm", "imm": "forbidden"},
    # "paddq_xx":        {"dst_reg": "required:xmm", "src_reg": "required:xmm", "imm": "forbidden"},
    # "pmuludq_xx":      {"dst_reg": "required:xmm", "src_reg": "required:xmm", "imm": "forbidden"},
}
# Exclude dst_reg == src_reg for opcodes where it is pointless (mov rax,rax no-op).
OPCODE_EXCLUDE_DST_EQ_SRC = ["mov_rr"]

# Registers forbidden in a given slot for CORRECTNESS (not style).
# div/idiv: when the divisor is RDX (== the high half of the implicit RDX:RAX
# dividend), the overflow-prevention instrumentation `D = (D & divisor) >> 1`
# is unsatisfiable, so check.py's sandbox_division() bails out (`return False`,
# see the `"rdx" in divisor.value` guard). The div is then left UN-sandboxed and
# faults (#DE) / is dropped at runtime. So rdx must never be the divisor.
OPCODE_FORBIDDEN_SRC_REGS = {
    "div_r":  {"rdx"},
    "idiv_r": {"rdx"},
}


def _apply_forbidden_src(opcode_name: str, mask: list) -> None:
    """Zero out src-reg slots forbidden for `opcode_name` (in place). mask index
    0 = empty, i+1 = SRC_REGS_SPACE[i]."""
    forbidden = OPCODE_FORBIDDEN_SRC_REGS.get(opcode_name)
    if not forbidden:
        return
    for i, name in enumerate(SRC_REGS_SPACE):
        if name in forbidden and (i + 1) < len(mask):
            mask[i + 1] = False


def _is_reg_empty(reg_id: int) -> bool:
    return reg_id == EMPTY_REG_ID


def _is_imm_empty(imm_id: int) -> bool:
    return imm_id == EMPTY_IMM_ID


def _parse_req(req):
    """
    Parse a reg-field requirement into (kind, klass).

    Accepted forms:
      "required"          -> ("required", "gpr")    # back-compat default
      "required:gpr"      -> ("required", "gpr")
      "required:xmm"      -> ("required", "xmm")
      "forbidden"         -> ("forbidden", None)
      "optional"          -> ("optional", "gpr")    # back-compat default
      "optional:xmm"      -> ("optional", "xmm")
    """
    if isinstance(req, str) and ":" in req:
        kind, klass = req.split(":", 1)
        return kind, klass
    if req == "forbidden":
        return "forbidden", None
    return req, "gpr"


def _reg_in_class(reg_id, klass):
    """
    True iff reg_id (1-based, 0=empty) refers to a register of the given class.
    GPR indices are 1..NUM_GPR; XMM indices are NUM_GPR+1..NUM_GPR+NUM_XMM.
    """
    if _is_reg_empty(reg_id):
        return False
    if klass == "gpr":
        return 1 <= reg_id <= NUM_GPR
    if klass == "xmm":
        return NUM_GPR + 1 <= reg_id <= NUM_GPR + NUM_XMM
    return True  # any


# Map 64-bit GPR name -> 32-bit sub-register name. Revizor's printer uses the
# RegisterOp's `value` field verbatim as the assembly token, so to emit `edx`
# instead of `rdx` we must pass the 32-bit name explicitly. base.json's `movd`
# only has REG32 forms; passing "rdx" with width=32 prints as `rdx` and the
# asm parser then has no matching spec.
_GPR64_TO_GPR32 = {
    "rax": "eax", "rbx": "ebx", "rcx": "ecx",
    "rdx": "edx", "rsi": "esi", "rdi": "edi",
}


def _to_gpr32(name):
    return _GPR64_TO_GPR32.get(name, name)


# 64-bit GPR name -> 8-bit (low byte) sub-register name, for setcc (writes a byte).
_GPR64_TO_GPR8 = {
    "rax": "al", "rbx": "bl", "rcx": "cl",
    "rdx": "dl", "rsi": "sil", "rdi": "dil",
}


def _to_gpr8(name):
    return _GPR64_TO_GPR8.get(name, name)


def _check_rule_operand_spec(opcode_id: int, reg_src_id: int, reg_dst_id: int, imm_id: int) -> bool:
    """Rule 1: OPCODE_OPERAND_SPEC (required/optional/forbidden + reg-class)."""
    if opcode_id < 1 or opcode_id > len(OPERAND_SPACE):
        return False
    opcode_name = OPERAND_SPACE[opcode_id - 1]
    spec = OPCODE_OPERAND_SPEC.get(opcode_name, {})
    dst_kind, dst_klass = _parse_req(spec.get("dst_reg", "optional"))
    src_kind, src_klass = _parse_req(spec.get("src_reg", "optional"))
    imm_req = spec.get("imm", "optional")

    def check_reg(kind: str, klass, reg_id: int) -> bool:
        if kind == "required":
            return (not _is_reg_empty(reg_id)) and _reg_in_class(reg_id, klass)
        if kind == "forbidden":
            return _is_reg_empty(reg_id)
        # optional: empty is fine; if a reg is chosen it must match the class
        if _is_reg_empty(reg_id):
            return True
        return _reg_in_class(reg_id, klass)

    def check_imm(field_req: str, is_empty: bool) -> bool:
        if field_req == "required":
            return not is_empty
        if field_req == "forbidden":
            return is_empty
        return True

    if not check_reg(dst_kind, dst_klass, reg_dst_id):
        return False
    if not check_reg(src_kind, src_klass, reg_src_id):
        return False
    if not check_imm(imm_req, _is_imm_empty(imm_id)):
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
    Also enforces reg-class (gpr / xmm) per OPCODE_OPERAND_SPEC.
    """
    if opcode_id == 0:
        return reg_src_id == 0 and reg_dst_id == 0 and imm_id == 0
    if not _check_rule_operand_spec(opcode_id, reg_src_id, reg_dst_id, imm_id):
        return False
    opcode_name = OPERAND_SPACE[opcode_id - 1]
    # Correctness: forbid specific src regs (e.g., div/idiv divisor == rdx is
    # unsandboxable, see OPCODE_FORBIDDEN_SRC_REGS).
    forbidden = OPCODE_FORBIDDEN_SRC_REGS.get(opcode_name)
    if forbidden and reg_src_id > 0 and SRC_REGS_SPACE[reg_src_id - 1] in forbidden:
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


def _class_slot_mask(klass: str, n_reg: int) -> list:
    """Return [bool] * (n_reg+1) with True only on indices belonging to `klass`."""
    out = [False] * (n_reg + 1)
    if klass == "gpr":
        for i in range(1, NUM_GPR + 1):
            out[i] = True
    elif klass == "xmm":
        for i in range(NUM_GPR + 1, NUM_GPR + NUM_XMM + 1):
            out[i] = True
    else:  # any
        for i in range(1, n_reg + 1):
            out[i] = True
    return out


def get_reg_src_mask(opcode_id: int) -> list:
    """Mask for reg_src given opcode_id. Length = n_regs + 1 (index 0 = empty)."""
    n_reg = len(SRC_REGS_SPACE)
    if opcode_id == 0:
        return [True] + [False] * n_reg
    opcode_name = OPERAND_SPACE[opcode_id - 1]
    spec = OPCODE_OPERAND_SPEC.get(opcode_name, {})
    kind, klass = _parse_req(spec.get("src_reg", "optional"))
    if kind == "required":
        cls = _class_slot_mask(klass, n_reg)
        cls[0] = False  # empty disallowed for required
        _apply_forbidden_src(opcode_name, cls)
        return cls
    if kind == "forbidden":
        return [True] + [False] * n_reg
    # optional
    cls = _class_slot_mask(klass, n_reg)
    cls[0] = True
    _apply_forbidden_src(opcode_name, cls)
    return cls


def get_reg_dst_mask(opcode_id: int, reg_src_id: int = 0) -> list:
    """Mask for reg_dst given opcode_id and reg_src_id (for OPCODE_EXCLUDE_DST_EQ_SRC)."""
    n_reg = len(DST_REGS_SPACE)
    if opcode_id == 0:
        return [True] + [False] * n_reg
    opcode_name = OPERAND_SPACE[opcode_id - 1]
    spec = OPCODE_OPERAND_SPEC.get(opcode_name, {})
    kind, klass = _parse_req(spec.get("dst_reg", "optional"))
    if kind == "required":
        base = _class_slot_mask(klass, n_reg)
        base[0] = False
    elif kind == "forbidden":
        base = [True] + [False] * n_reg
    else:  # optional
        base = _class_slot_mask(klass, n_reg)
        base[0] = True
    # Exclude dst==src for mov_rr (etc.) when the src reg is in the same class.
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

    # ---- new pure-GPR opcodes ----
    # BASE-BINARY: div/idiv reg64. RAX:RDX is implicit dst, only src_reg
    # carries the divisor register.
    if opcode == "div_r":
        return Instruction("div", False, "", False).add_op(
            RegisterOp(rs, 64, True, False)
        )

    if opcode == "idiv_r":
        return Instruction("idiv", False, "", False).add_op(
            RegisterOp(rs, 64, True, False)
        )

    # BASE-SEMAPHORE: atomic RMW on [reg], reg.  dst_reg is the mem base,
    # src_reg is the value register.  Revizor encodes "lock <op>" as the
    # instruction name (literal lowercase, space-separated prefix).
    if opcode == "lock_xadd_mr":
        return Instruction("lock xadd", False, "", False).add_op(
            MemoryOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, True))

    if opcode == "lock_xchg_mr":
        # `xchg` on a memory operand is implicitly atomic per ISA; Revizor's
        # base.json does not have a "lock xchg" entry — use plain `xchg`,
        # and _sandbox_memory_access still applies the 8-byte alignment mask
        # via `if "lock" in instr.name or instr.name == "xchg"` (generator.py).
        return Instruction("xchg", False, "", False).add_op(
            MemoryOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, True))

    if opcode == "lock_cmpxchg_mr":
        # cmpxchg also implicitly reads/writes RAX as the comparand.  We do
        # not surface that in the action space; the sandbox pass and ISA spec
        # handle the implicit operand.
        return Instruction("lock cmpxchg", False, "", False).add_op(
            MemoryOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "lock_add_mr":
        return Instruction("lock add", False, "", False).add_op(
            MemoryOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    # BASE-STRINGOP: rep-prefixed string ops.  No explicit operands — RSI,
    # RDI, RCX, AL are all implicit.  Revizor's _sandbox_repeated_instruction
    # pass clamps RCX to 0..255 before the rep, and _sandbox_memory_access
    # already handles implicit memory operands for STRINGOP via has_mem_operand(True).
    # We use the `repe` prefix uniformly; for movsb/stosb (no flag check)
    # repe / repne / rep all behave the same in hardware.
    if opcode == "rep_movsb":
        return Instruction("repe movsb", False, "", False)

    if opcode == "rep_stosb":
        return Instruction("repe stosb", False, "", False)

    if opcode == "rep_cmpsb":
        return Instruction("repe cmpsb", False, "", False)

    if opcode == "rep_scasb":
        return Instruction("repe scasb", False, "", False)

    # BASE-CMOV: cmov reg64, [reg64]. dst_reg is the destination GPR (R/W:
    # cmov reads the prior value when the condition is false), src_reg is
    # the memory base.
    # cmova == cmovnbe (same opcode); base.json carries the cmovnbe name.
    _CMOV_RM_NAMES = {
        "cmovz_rm":  "cmovz",
        "cmovnz_rm": "cmovnz",
        "cmovb_rm":  "cmovb",
        "cmova_rm":  "cmovnbe",
        "cmovbe_rm":  "cmovbe",
        "cmovl_rm":   "cmovl",
        "cmovle_rm":  "cmovle",
        "cmovnl_rm":  "cmovnl",
        "cmovnle_rm": "cmovnle",
        "cmovs_rm":   "cmovs",
        "cmovns_rm":  "cmovns",
    }
    if opcode in _CMOV_RM_NAMES:
        return Instruction(_CMOV_RM_NAMES[opcode], False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(MemoryOp(rs, 64, True, False))

    # BASE-BINARY: sub / sbb reg64, reg64.
    if opcode == "sub_rr":
        return Instruction("sub", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "sbb_rr":
        return Instruction("sbb", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    # ---- NEW GPR opcodes ----
    # cmp/test: flag producers, dst is read-only (not written).
    if opcode == "cmp_rr":
        return Instruction("cmp", False, "", False).add_op(
            RegisterOp(rd, 64, True, False)
        ).add_op(RegisterOp(rs, 64, True, False))
    if opcode == "cmp_rm":
        return Instruction("cmp", False, "", False).add_op(
            RegisterOp(rd, 64, True, False)
        ).add_op(MemoryOp(rs, 64, True, False))
    if opcode == "cmp_mr":
        return Instruction("cmp", False, "", False).add_op(
            MemoryOp(rd, 64, True, False)
        ).add_op(RegisterOp(rs, 64, True, False))
    if opcode == "test_rr":
        return Instruction("test", False, "", False).add_op(
            RegisterOp(rd, 64, True, False)
        ).add_op(RegisterOp(rs, 64, True, False))
    # and/or/adc/imul reg64, reg64 : dst is read+write.
    if opcode == "and_rr":
        return Instruction("and", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))
    if opcode == "or_rr":
        return Instruction("or", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))
    if opcode == "adc_rr":
        return Instruction("adc", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))
    if opcode == "imul_rr":
        return Instruction("imul", False, "", False).add_op(
            RegisterOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))
    # unary reg64 : dst read+write.
    if opcode == "not_r":
        return Instruction("not", False, "", False).add_op(RegisterOp(rd, 64, True, True))
    if opcode == "neg_r":
        return Instruction("neg", False, "", False).add_op(RegisterOp(rd, 64, True, True))
    if opcode == "inc_r":
        return Instruction("inc", False, "", False).add_op(RegisterOp(rd, 64, True, True))
    if opcode == "dec_r":
        return Instruction("dec", False, "", False).add_op(RegisterOp(rd, 64, True, True))
    # setcc reg8 : writes the low byte of a GPR (partial-register write).
    _SETCC_NAMES = {
        "setz_r": "setz", "setnz_r": "setnz", "setb_r": "setb", "setl_r": "setl",
    }
    if opcode in _SETCC_NAMES:
        return Instruction(_SETCC_NAMES[opcode], False, "", False).add_op(
            RegisterOp(_to_gpr8(rd), 8, False, True)
        )
    # movzx/movsx reg64, byte ptr [reg64] : zero/sign-extend a byte load.
    if opcode == "movzx_rm":
        return Instruction("movzx", False, "", False).add_op(
            RegisterOp(rd, 64, False, True)
        ).add_op(MemoryOp(rs, 8, True, False))
    if opcode == "movsx_rm":
        return Instruction("movsx", False, "", False).add_op(
            RegisterOp(rd, 64, False, True)
        ).add_op(MemoryOp(rs, 8, True, False))
    # lock <op> [reg64], reg64 : atomic RMW, dst=mem base, src=value reg.
    _LOCK_MR_NAMES = {
        "lock_sub_mr": "lock sub", "lock_and_mr": "lock and",
        "lock_or_mr": "lock or", "lock_xor_mr": "lock xor",
    }
    if opcode in _LOCK_MR_NAMES:
        return Instruction(_LOCK_MR_NAMES[opcode], False, "", False).add_op(
            MemoryOp(rd, 64, True, True)
        ).add_op(RegisterOp(rs, 64, True, False))
    # lock inc/dec [reg64] : single mem operand RMW.
    if opcode == "lock_inc_m":
        return Instruction("lock inc", False, "", False).add_op(MemoryOp(rd, 64, True, True))
    if opcode == "lock_dec_m":
        return Instruction("lock dec", False, "", False).add_op(MemoryOp(rd, 64, True, True))

    # BASE-LOGICAL atomic: lock not [reg64]. Single mem operand.
    if opcode == "lock_not_m":
        return Instruction("lock not", False, "", False).add_op(
            MemoryOp(rd, 64, True, True)
        )

    # ---- XMM (SSE / SSE2) ----
    # By the mask spec above, `rd` and `rs` are guaranteed to be in the
    # correct class — e.g., for movdqu_xm: rd is xmm*, rs is r{ax,bx,...}.
    if opcode == "movdqu_xm":
        # movdqu xmm, [reg64]   -- base.json: SSE2-DATAXFER movdqu REG128, MEM128
        return Instruction("movdqu", False, "", False).add_op(
            RegisterOp(rd, 128, False, True)
        ).add_op(MemoryOp(rs, 128, True, False))

    if opcode == "movdqu_mx":
        # movdqu [reg64], xmm   -- base.json: movdqu MEM128, REG128
        return Instruction("movdqu", False, "", False).add_op(
            MemoryOp(rd, 128, False, True)
        ).add_op(RegisterOp(rs, 128, True, False))

    if opcode == "movups_xm":
        # movups xmm, [reg64]   -- base.json: SSE-DATAXFER movups REG128, MEM128
        return Instruction("movups", False, "", False).add_op(
            RegisterOp(rd, 128, False, True)
        ).add_op(MemoryOp(rs, 128, True, False))

    if opcode == "movups_mx":
        # movups [reg64], xmm   -- base.json: movups MEM128, REG128
        return Instruction("movups", False, "", False).add_op(
            MemoryOp(rd, 128, False, True)
        ).add_op(RegisterOp(rs, 128, True, False))

    if opcode == "movq_xr":
        # movq xmm, reg64       -- base.json: SSE2-DATAXFER movq REG128, REG64
        return Instruction("movq", False, "", False).add_op(
            RegisterOp(rd, 128, False, True)
        ).add_op(RegisterOp(rs, 64, True, False))

    if opcode == "movd_xr":
        # movd xmm, reg32       -- base.json: movd REG128, REG32 (GPR low 32 -> XMM).
        # GPR side must use the 32-bit sub-name (e.g. `edx`, not `rdx`);
        # Revizor's printer emits the RegisterOp.value verbatim, and the asm
        # parser will reject `movd xmm, rdx` ("no matching spec").
        return Instruction("movd", False, "", False).add_op(
            RegisterOp(rd, 128, False, True)
        ).add_op(RegisterOp(_to_gpr32(rs), 32, True, False))

    # movd_rx removed: base.json lacks the `movd (REG32,REG128)` (reg32 <- xmm)
    # form, so it can't be encoded without mis-typing the XMM operand as a GPR.

    if opcode == "pxor_xx":
        # pxor xmm, xmm         -- base.json: SSE2-LOGICAL pxor REG128, REG128
        return Instruction("pxor", False, "", False).add_op(
            RegisterOp(rd, 128, True, True)
        ).add_op(RegisterOp(rs, 128, True, False))

    if opcode == "paddq_xx":
        # paddq xmm, xmm        -- base.json: SSE2-SSE paddq REG128, REG128
        return Instruction("paddq", False, "", False).add_op(
            RegisterOp(rd, 128, True, True)
        ).add_op(RegisterOp(rs, 128, True, False))

    if opcode == "pmuludq_xx":
        # pmuludq xmm, xmm      -- base.json: SSE2-SSE pmuludq REG128, REG128
        return Instruction("pmuludq", False, "", False).add_op(
            RegisterOp(rd, 128, True, True)
        ).add_op(RegisterOp(rs, 128, True, False))

    return None


# Backward compatibility aliases
Operand_Space = OPERAND_SPACE
Dst_Regs_Space = DST_REGS_SPACE
Src_Regs_Space = SRC_REGS_SPACE
Imms_Space = IMMS_SPACE
