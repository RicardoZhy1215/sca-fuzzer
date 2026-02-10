.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
.line_1:
MOV rbx, rcx 
lfence  
.line_2:
XOR rdx, rcx 
lfence  
.line_3:
AND rcx, 0b1111111111111 # instrumentation
lfence  
SBB qword ptr [R14 + rcx], 35 
lfence  
.line_4:
AND rcx, 0b1111111111111 # instrumentation
lfence  
IMUL byte ptr [R14 + rcx] 
lfence  
.line_5:
lfence
.line_6:
lfence
.line_7:
lfence
.line_8:
lfence
.line_9:
lfence
.line_10:
lfence
.line_11:
lfence
.line_12:
lfence
.line_13:
lfence
.line_14:
lfence
.line_15:
lfence
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
