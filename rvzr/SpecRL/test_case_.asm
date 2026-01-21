.intel_syntax noprefix
MFENCE # instrumentation
.test_case_enter:
.line_1:
AND RAX, 0b1111111111111 # instrumentation
SBB qword ptr [R14 + RAX], 35
.line_2:
JMP .line_4 
.line_3:
JMP .line_10 
.line_4:
AND RBX, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + RBX] 
.line_5:
JNS .line_7
.line_6:
AND RCX, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + RCX]
.line_7:
AND RAX, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + RAX]
.line_8:
AND RBX, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + RBX]
.line_9:
JNS line_10
.line_11:
.line_12:
.line_13:
.line_14:
.line_15:
.test_case_exit:
MFENCE # instrumentation
