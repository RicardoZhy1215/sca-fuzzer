.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
SBB qword ptr [R14 + rcx], 35 
JNS .line_6 # instrumentation
XOR rcx, rbx 
XOR rcx, rcx 
MOV rcx, rdx 
and rbx, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + rbx] 
XOR rcx, rax 
XOR rcx, rax 
and rdx, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + rdx] 
JMP .line_10 # instrumentation
XOR rdx, rdx 
and rbx, 0b1111111111111 # instrumentation
SBB qword ptr [R14 + rbx], 35 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
