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
ADD_RDI_RSI = Instruction("add", False, "", False) \
    .add_op(RegisterOp("rdi", 64, False, True)) \
    .add_op(MemoryOp("rsi", 64, True, False))

ADD_CL_DL = Instruction("add", False, "", False) \
    .add_op(RegisterOp("cl", 8, False, True)) \
    .add_op(RegisterOp("dl", 8, True, False))

ADD_RCX_RBX = Instruction("add", False, "", False) \
    .add_op(MemoryOp("rcx", 64, False, True)) \
    .add_op(RegisterOp("rbx", 64, True, False))

ADD_RBX_ECX = Instruction("add", False, "", False) \
    .add_op(MemoryOp("rbx", 32, False, True)) \
    .add_op(RegisterOp("ecx", 32, True, False))

CMP_RAX_ECX = Instruction("cmp", False, "", False) \
    .add_op(MemoryOp("rax", 32, False, True)) \
    .add_op(RegisterOp("ecx", 32, True, False))

DIV_RDI = Instruction("div", False, "", False) \
    .add_op(MemoryOp("rdi", 8, True, False))

SUB_RSI_BL = Instruction("sub", False, "", False) \
    .add_op(MemoryOp("rsi", 8, False, True)) \
    .add_op(RegisterOp("bl", 8, True, False))

SUB_AL_RCX = Instruction("sub", False, "", False) \
    .add_op(RegisterOp("al", 8, False, True)) \
    .add_op(MemoryOp("rcx", 8, True, False))

MUL_RCX = Instruction("mul", False, "", False) \
    .add_op(MemoryOp("rcx", 64, True, False))

SUB_RAX_128 = Instruction("lock sub", False, "", False) \
    .add_op(MemoryOp("rax", 8, False, True)) \
    .add_op(ImmediateOp("-128", 8))


# test_instruction_space = [SBB_qword_ptr_R14_RBX_35, JNS_line_4, JMP_line_5, IMUL_byte_ptr_R14_RCX, MOV_RCX_R14, MOV_RAX_RCX, MOV_RCX_10, ADD_RCX_5]
test_instruction_space = [ADD_RDI_RSI, ADD_CL_DL, ADD_RCX_RBX, ADD_RBX_ECX, CMP_RAX_ECX, DIV_RDI, SUB_RSI_BL, SUB_AL_RCX, MUL_RCX, SUB_RAX_128]

env_config = {"instruction_space": test_instruction_space,
              "sequence_size": 15,
              "num_inputs": 1}

if __name__ == "__main__":
    test = SpecEnv(env_config)
    test.step(0)  # ADD RDI, [RSI]
    test.step(1)  # ADD CL, DL
    test.step(2)  # ADD [RCX], RBX
    test.step(3)  # ADD [RBX], ECX  
    test.step(4)  # CMP [RAX], ECX
    test.step(5)  # DIV [RDI]
    test.step(6)  # SUB [RSI], BL
    test.step(7)  # SUB AL, [RCX]
    test.step(8)  # MUL [RCX]
    test.step(9)  # LOCK SUB [RAX], -128

    # print("MOV RCX, 10")
    # test.step(6)
    # print("MOV RCX, R14")
    # test.step(4)
    # print("ADD RCX, 5")
    # test.step(7)
    # print("MOV RAX, [RCX]")
    # test.step(5)
    exit()
