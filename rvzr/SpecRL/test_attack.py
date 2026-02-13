from SpecEnv import SpecEnv
from rvzr.tc_components.instruction import Instruction, RegisterOp, ImmediateOp, MemoryOp, LabelOp

SBB_qword_ptr_R14_RBX_35 = Instruction("SBB", False, "",  False) \
    .add_op(MemoryOp("RBX", 64, True, True)) \
    .add_op(ImmediateOp("35", 8))
IMUL_byte_ptr_R14_RCX = Instruction("IMUL", False, "",  False) \
    .add_op(MemoryOp("RCX", 8, True, False)) \
    .add_op(RegisterOp("RAX", 64, True, True), implicit=True)
JNS_line_4 = Instruction("JNS", False, "", True) \
    .add_op(LabelOp(".line_4"))
JMP_line_5 = Instruction("JMP", False, "", True) \
    .add_op(LabelOp(".line_5"))
MOV_RCX_R14 = Instruction("MOV", False, "", False) \
    .add_op(RegisterOp("RCX", 64, False, True)) \
    .add_op(RegisterOp("R14", 64, True, False))
MOV_RAX_RCX = Instruction("MOV", False, "", False) \
    .add_op(RegisterOp("RAX", 64, False, True)) \
    .add_op(MemoryOp("RCX", 64, True, False))
MOV_RCX_10 = Instruction("MOV", False, "", False) \
    .add_op(RegisterOp("RCX", 64, False, False)) \
    .add_op(ImmediateOp("10", 8))
ADD_RCX_5 = Instruction("ADD", False, "", False) \
    .add_op(RegisterOp("RCX", 64, False, True)) \
    .add_op(ImmediateOp("5", 8))



# Attack instructions sequence based on revizor example 
DEC_DI  = Instruction("dec", False, "", False) \
    .add_op(RegisterOp("di", 16, False, True))

AND_RDX_0B = Instruction("and", False, "", is_instrumentation=True) \
    .add_op(RegisterOp("rdx", 64, False, True)) \
    .add_op(ImmediateOp("0b1111111111000", 8))

OR_RDX_0B = Instruction("or", False, "", is_instrumentation=True) \
    .add_op(MemoryOp("r14 + rdx", 16, False, True)) \
    .add_op(ImmediateOp("0b1000", 16))

AND_RDX_0B_MEM = Instruction("and", False, "", is_instrumentation=True) \
    .add_op(MemoryOp("r14 + rdx", 8, False, True)) \
    .add_op(ImmediateOp("0b1111111111000", 8))

AND_RBX_0B = Instruction("and", False, "", is_instrumentation=True) \
    .add_op(RegisterOp("rbx", 64, False, True)) \
    .add_op(ImmediateOp("0b1111111111000", 64))

AND_RAX_0B = Instruction("and", False, "", is_instrumentation=True) \
    .add_op(RegisterOp("rax", 32, False, True)) \
    .add_op(ImmediateOp("0b1111111111000", 32))

MUL_RAX = Instruction("mul", False, "", False) \
    .add_op(MemoryOp("r14 + rax", 32, False, True)) \
    
AND_RDI_0B = Instruction("and", False, "", is_instrumentation=True) \
    .add_op(RegisterOp("rdi", 32, False, True)) \
    .add_op(ImmediateOp("0b1111111111111", 32))
MUL_RDI = Instruction("mul", False, "", False) \
    .add_op(MemoryOp("r14 + rdi", 32, False, True)) \

ADD_AL_110 = Instruction("add", False, "", False) \
    .add_op(RegisterOp("al", 8, False, True)) \
    .add_op(ImmediateOp("-110", 8))

JBE_BB_01 = Instruction("jbe", True, "", False) \
    .add_op(LabelOp(".bb_0.1"))

JMP_EXIT = Instruction("jmp", True, "", False) \
    .add_op(LabelOp(".exit_0"))



MOV_RBX_1 = Instruction("mov", False, "", False) \
    .add_op(MemoryOp("r14 + rbx", 64, False, True)) \
    .add_op(ImmediateOp("1", 64))




# test_instruction_space = [SBB_qword_ptr_R14_RBX_35, JNS_line_4, JMP_line_5, IMUL_byte_ptr_R14_RCX, MOV_RCX_R14, MOV_RAX_RCX, MOV_RCX_10, ADD_RCX_5]
test_instruction_space = [DEC_DI, AND_RDX_0B, OR_RDX_0B, AND_RDX_0B_MEM, AND_RBX_0B, AND_RAX_0B,MUL_RAX,AND_RDI_0B, MUL_RDI, ADD_AL_110, JBE_BB_01, JMP_EXIT, AND_RBX_0B, MOV_RBX_1]

env_config = {"instruction_space": test_instruction_space,
              "sequence_size": 15,
              "num_inputs": 1}

if __name__ == "__main__":
    test = SpecEnv(env_config)
    test.step(0)  
    test.step(1)  
    test.step(2) 
    test.step(3)  
    test.step(4)  
    test.step(5) 
    test.step(6)  
    test.step(7) 
    test.step(8) 
    test.step(9)  
    test.step(10) 
    test.step(11)
    test.step(12)
    test.step(13)

    # print("MOV RCX, 10")
    # test.step(6)
    # print("MOV RCX, R14")
    # test.step(4)
    # print("ADD RCX, 5")
    # test.step(7)
    # print("MOV RAX, [RCX]")
    # test.step(5)
    exit()
