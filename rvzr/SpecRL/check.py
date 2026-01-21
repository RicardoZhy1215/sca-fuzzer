import math
import re
import random
from typing import List, Dict, Set, Optional, Tuple


from rvzr.tc_components.test_case_code import Program
from rvzr.tc_components.instruction import Instruction
from rvzr.tc_components.instruction import Operand,RegisterOp, MemoryOp, ImmediateOp, FlagsOp

from rvzr.config import CONF
from rvzr.arch.x86.target_desc import X86TargetDesc
import copy

def X86NonCanonicalAddressCheck(prog: Program, instr: Instruction) -> bool:
    if 'GP-noncanonical' not in CONF.permitted_faults:
        return True
    
    if not instr.has_mem_operand(True):
        return True
    
    src_operands = []
    for o in instr.get_src_operands():
        if isinstance(o, RegisterOp):
            src_operands.append(o)

    mem_operands = instr.get_mem_operands()
    implicit_mem_operands = instr.get_implicit_mem_operands()
    if mem_operands and not implicit_mem_operands:
        assert len(mem_operands) == 1, f"Unexpected instruction format {instr.name}"
        mem_operand: Operand = mem_operands[0]
        registers = mem_operand.value

        masks_list = ["RAX", "RBX"]
        mask_reg = masks_list[0]
        # Do not overwrite offset register with mask
        for operands in src_operands:
            op_regs = re.split(r'\+|-|\*| ', operands.value)
            for reg in op_regs:
                if X86TargetDesc.reg_normalized[mask_reg] == \
                    X86TargetDesc.reg_normalized[reg]:
                    mask_reg = masks_list[1]

        offset_list = ["RCX", "RDX"]
        offset_reg = offset_list[0]
        # Do not reuse destination register
        for op in instr.get_all_operands():
            if not isinstance(op, RegisterOp):
                continue
            if X86TargetDesc.reg_normalized[offset_reg] == \
                X86TargetDesc.reg_normalized[op.value]:
                offset_reg = offset_list[1]

        mask = hex((random.getrandbits(16) << 48))
        lea = Instruction("LEA", True) \
            .add_op(RegisterOp(offset_reg, 64, False, True)) \
            .add_op(MemoryOp(registers, 64, True, False))
        prog.append(lea)
        mov = Instruction("MOV", True) \
            .add_op(RegisterOp(mask_reg, 64, True, True)) \
            .add_op(ImmediateOp(mask, 64))
        prog.append(mov)
        mask = Instruction("XOR", True) \
            .add_op(RegisterOp(offset_reg, 64, True, True)) \
            .add_op(RegisterOp(mask_reg, 64, True, False))
        prog.append(mask)
        for idx, op in enumerate(instr.operands):
            if op == mem_operand:
                old_op = instr.operands[idx]
                addr_op = MemoryOp(offset_reg, old_op.get_width(),
                                        old_op.src, old_op.dest)
                instr.operands[idx] = addr_op
    return True

def X86SandboxCheck(prog: Program, instr: Instruction, target_desc: X86TargetDesc) -> bool:
    mask_3bits = "0b111"
    bit_test_names = ["BT", "BTC", "BTR", "BTS", "LOCK BT", "LOCK BTC", "LOCK BTR", "LOCK BTS"]
    input_memory_size = CONF.input_main_region_size + CONF.input_faulty_region_size
    mask_size = int(math.log(input_memory_size, 2)) - CONF.memory_access_zeroed_bits
    sandbox_address_mask = "0b" + "1" * mask_size + "0" * CONF.memory_access_zeroed_bits

    def sandbox_memory_access(instr: Instruction, prog: Program) -> bool:
        """ Force the memory accesses into the page starting from R14 """
        mem_operands = instr.get_mem_operands()
        implicit_mem_operands = instr.get_implicit_mem_operands()
        if mem_operands and not implicit_mem_operands:
            assert len(mem_operands) == 1, \
                f"Instructions with multiple memory accesses are not yet supported: {instr.name}"
            mem_operand: Operand = mem_operands[0]
            address_reg = mem_operand.value
            if instr.instrumented: address_reg = address_reg[6:]
            imm_width = mem_operand.width if mem_operand.width <= 32 else 32
            apply_mask = Instruction("AND", True) \
                .add_op(RegisterOp(address_reg, mem_operand.width, True, True)) \
                .add_op(ImmediateOp(sandbox_address_mask, imm_width)) \
                .add_op(FlagsOp(["w", "w", "undef", "w", "w", "", "", "", "w"]), True)
            prog.append(apply_mask)
            # prevent double adding R14
            if not instr.instrumented:
                instr.get_mem_operands()[0].value = "R14 + " + address_reg
                instr.instrumented = True
            return True

        mem_operands = implicit_mem_operands
        if mem_operands:
            # deduplicate operands
            uniq_operands: Dict[str, MemoryOp] = {}
            for o in mem_operands:
                if o.value not in uniq_operands:
                    uniq_operands[o.value] = o

            # instrument each operand to sandbox the memory accesses
            for address_reg, mem_operand in uniq_operands.items():
                imm_width = mem_operand.width if mem_operand.width <= 32 else 32
                assert address_reg in target_desc.registers_by_size[64], \
                    f"Unexpected address register {address_reg} used in {instr}"
                apply_mask = Instruction("AND", True) \
                    .add_op(RegisterOp(address_reg, mem_operand.width, True, True)) \
                    .add_op(ImmediateOp(sandbox_address_mask, imm_width)) \
                    .add_op(FlagsOp(["w", "w", "undef", "w", "w", "", "", "", "w"]), True)
                add_base = Instruction("ADD", True) \
                    .add_op(RegisterOp(address_reg, mem_operand.width, True, True)) \
                    .add_op(RegisterOp("R14", 64, True, False)) \
                    .add_op(FlagsOp(["w", "w", "undef", "w", "w", "", "", "", "w"]), True)
                prog.append(apply_mask)
                prog.append(add_base)
            return True
        return False

    def sandbox_division(instr: Instruction, prog: Program) -> bool:
        """
        We do not support handling of division faults so far, so we have to prevent them.
        Specifically, we need to prevent two types of faults:
        - division by zero
        - division overflow (i.e., quotient is larger than the destination register)
        For this, we ensure that the *D register (upper half of the dividend) is always
        less than the divisor with a bit trick like this ( D & divisor >> 1).

        The first corner case when it won't work is when the divisor is D. This case
        is impossible to resolve, as far as I can tell. We just give up.

        The second corner case is 8-bit division, when the divisor is the AX register alone.
        Here the instrumentation become too complicated, and we simply set AX to 1.
        """
        divisor = instr.operands[0]

        if divisor.width == 64 and CONF.x86_disable_div64:  # type: ignore
            return False

        if 'DE-zero' not in CONF.permitted_faults:
            # Prevent div by zero
            instrumentation = Instruction("OR", True) \
                .add_op(divisor) \
                .add_op(ImmediateOp("1", 8)) \
                .add_op(FlagsOp(["w", "w", "undef", "w", "w", "", "", "", "w"]), True)
            prog.append(instrumentation)

        if 'DE-overflow' in CONF.permitted_faults:
            return False

        # divisor in D or in memory with RDX offset? Impossible case, give up
        if divisor.value in ["RDX", "EDX", "DX", "DH", "DL"] or "RDX" in divisor.value:
            return False

        # dividend in AX?
        if divisor.width == 8:
            if "RAX" not in divisor.value:
                instrumentation = Instruction("MOV", True).\
                    add_op(RegisterOp("AX", 16, False, True)).\
                    add_op(ImmediateOp("1", 16))
                prog.append(instrumentation)
                return True
            else:
                # AX is both the dividend and the offset in memory.
                # Too complex (impossible?). Giving up
                return False

        # Normal case
        # D = (D & divisor) >> 1
        d_register = {64: "RDX", 32: "EDX", 16: "DX"}[divisor.width]
        instrumentation = Instruction("AND", True) \
            .add_op(RegisterOp(d_register, divisor.width, False, True)) \
            .add_op(divisor) \
            .add_op(FlagsOp(["w", "w", "undef", "w", "w", "", "", "", "w"]), True)
        prog.append(instrumentation)
        instrumentation = Instruction("SHR", True) \
            .add_op(RegisterOp(d_register, divisor.width, False, True)) \
            .add_op(ImmediateOp("1", 8)) \
            .add_op(FlagsOp(["w", "w", "undef", "w", "w", "", "", "", "undef"]), True)
        prog.append(instrumentation)

        return True
    
    def sandbox_bit_test(instr: Instruction, prog: Program) -> bool:
        """
        The address accessed by a BT* instruction is based on both of its operands.
        `sandbox_memory_access` take care of the first operand.
        This function ensures that the offset is always within a byte.
        """
        address = instr.operands[0]
        if isinstance(address, RegisterOp):
            # this is a version that does not access memory
            # no need for sandboxing
            return True

        offset = instr.operands[1]
        if isinstance(offset, ImmediateOp):
            # The offset is an immediate
            # Simply replace it with a smaller value
            offset.value = str(random.randint(0, 7))
            return True

        # The offset is in a register
        # Mask its upper bits to reduce the stored value to at most 7
        if address.value != offset.value:
            apply_mask = Instruction("AND", True) \
                .add_op(offset) \
                .add_op(ImmediateOp(mask_3bits, 8)) \
                .add_op(FlagsOp(["w", "w", "undef", "w", "w", "", "", "", "w"]), True)
            prog.append(apply_mask)
            return True

        # Special case: offset and address use the same register
        # Sandboxing is impossible. Give up
        return False
    
    def sandbox_repeated_instruction(instr: Instruction, prog: Program) -> bool:
        apply_mask = Instruction("AND", True) \
            .add_op(RegisterOp("RCX", 64, True, True)) \
            .add_op(ImmediateOp("0xff", 8)) \
            .add_op(FlagsOp(["w", "w", "undef", "w", "w", "", "", "", "w"]), True)
        add_base = Instruction("ADD", True) \
            .add_op(RegisterOp("RCX", 64, True, True)) \
            .add_op(ImmediateOp("1", 1)) \
            .add_op(FlagsOp(["w", "w", "w", "w", "w", "", "", "", "w"]), True)
        prog.append(apply_mask)
        prog.append(add_base)

        return True

    def sandbox_corrupted_cf(instr: Instruction, prog: Program) -> bool:
        set_cf = Instruction("STC", True) \
            .add_op(FlagsOp(["w", "", "", "", "", "", "", "", ""]), True)
        prog.append(set_cf)

        return True

    def sandbox_enclu(instr: Instruction, prog: Program) -> bool:
        options = [
            "0",  # ereport
            "1",  # egetkey
            "4",  # eexit
            "5",  # eaccept
            "6",  # emodpe
            "7",  # eacceptcopy
        ]
        set_rax = Instruction("MOV", True) \
            .add_op(RegisterOp("EAX", 32, True, True)) \
            .add_op(ImmediateOp(random.choice(options), 1))
        prog.append(set_rax)

        return True
    
    passed = True

    if instr.has_mem_operand(True):
        passed = sandbox_memory_access(instr, prog)
    if instr.name in ["DIV", "REX DIV"]:
        passed = sandbox_division(instr, prog)
    elif instr.name in bit_test_names:
        passed = sandbox_bit_test(instr, prog)
    elif "REP" in instr.name:
        passed = sandbox_repeated_instruction(instr, prog)
    elif instr.category == "BASE-ROTATE" or instr.category == "BASE-SHIFT":
        passed = sandbox_corrupted_cf(instr, prog)
    elif instr.name == "ENCLU":
        passed = sandbox_enclu(instr, prog)
    
    return passed

def X86PatchOpcodesCheck(prog: Program, instr: Instruction) -> bool:
    """
    Replaces assembly instructions with their opcodes.
    This is necessary to test instruction with multiple opcodes and
    the instruction that are not supported/not permitted by the standard
    assembler.
    """
    opcodes: Dict[str, List[str]] = {
        "UD2": [
            # UD2 instruction
            "0x0f, 0x0b",

            # invalid in 64-bit mode;
            # all the following opcodes are padded
            # with NOP to prevent misinterpretation by objdump
            "0x06, 0x90",  # 32-bit encoding of PUSH
            "0x07, 0x90",  # 32-bit encoding of POP
            "0x0E, 0x90",  # alternative 32-bit encoding of PUSH
            "0x16, 0x90",  # alternative 32-bit encoding of PUSH
            "0x17, 0x90",  # alternative 32-bit encoding of POP
            "0x1E, 0x90",  # alternative 32-bit encoding of PUSH
            "0x1F, 0x90",  # alternative 32-bit encoding of POP
            "0x27, 0x90",  # DAA
            "0x2F, 0x90",  # DAS
            "0x37, 0x90",  # AAA
            "0x3f, 0x90",  # AAS
            "0x60, 0x90",  # PUSHA
            "0x61, 0x90",  # POPA
            "0x62, 0x90",  # BOUND
            "0x82, 0x90",  # 32-bit aliases for logical instructions
            "0x9A, 0x90",  # 32-bit encoding of CALLF
            "0xC4, 0x90",  # LES
            "0xD4, 0x90",  # AAM
            "0xD5, 0x90",  # AAD
            "0xD6, 0x90",  # reserved
            "0xEA, 0x90",  # 32-bit encoding of JMPF
        ],
        "INT1": ["0xf1"]
    }

    if instr.name in opcodes.keys():
        opcode_options = opcodes[instr.name]
        opcode = random.choice(opcode_options)
        instr.name = ".byte " + opcode
    
    return True

def X86CheckAll(prog: Program, instr: Instruction, target_desc: X86TargetDesc) -> bool:
    return X86NonCanonicalAddressCheck(prog, instr) and X86PatchOpcodesCheck(prog, instr) and X86SandboxCheck(prog, instr, target_desc)
