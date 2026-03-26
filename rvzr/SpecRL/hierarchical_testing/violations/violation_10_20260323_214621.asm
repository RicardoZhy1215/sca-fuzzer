.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -87 # instrumentation
cmp rax, rsi 
xor rdi, rsi 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx 
jbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
