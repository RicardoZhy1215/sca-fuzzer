.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
lfence  
SBB qword ptr [R14 + rcx], 35 
lfence  
JMP .line_5 # instrumentation
lfence  
JMP .line_10 # instrumentation
lfence  
JNS .line_5 # instrumentation
lfence  
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
