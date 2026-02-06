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

test_instruction_space = [SBB_qword_ptr_R14_RBX_35, JNS_line_4, JMP_line_5, IMUL_byte_ptr_R14_RCX, MOV_RCX_R14, MOV_RAX_RCX, MOV_RCX_10, ADD_RCX_5]

env_config = {"instruction_space": test_instruction_space,
              "sequence_size": 15,
              "num_inputs": 1}

if __name__ == "__main__":
    test = SpecEnv(env_config)
    print("MOV RCX, 10")
    test.step(6)
    print("MOV RCX, R14")
    test.step(4)
    print("ADD RCX, 5")
    test.step(7)
    print("MOV RAX, [RCX]")
    test.step(5)
    exit()
