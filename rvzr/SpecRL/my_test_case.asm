.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
XOR rcx, rcx 
XOR rax, rcx 
JNS .line_4 # instrumentation
and rcx, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + rcx] 
and rax, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + rax] 
MOV rcx, rdx 
and rdx, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + rdx] 
JNS .line_1 # instrumentation
XOR rdx, rax 
MOV rax, rdx 
XOR rdx, rax 
MOV rbx, rax 
XOR rax, rax 
MOV rcx, rax 
and r14 + rdx, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
