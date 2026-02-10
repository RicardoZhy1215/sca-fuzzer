.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
.line_1:
CMP rbx, rcx 
lfence  
.line_2:
MOV rax, rcx 
lfence  
.line_3:
XOR rbx, rcx 
lfence  
.line_4:
AND rbx, 0b1111111111111 # instrumentation
lfence  
IMUL byte ptr [R14 + rbx] 
lfence  
.line_5:
JNS .line_5 # instrumentation
lfence  
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
