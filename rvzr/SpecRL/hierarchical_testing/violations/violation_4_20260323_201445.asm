.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -31 # instrumentation
add rax, rdi 
add rdi, rdx 
jns .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rsi, rbx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
