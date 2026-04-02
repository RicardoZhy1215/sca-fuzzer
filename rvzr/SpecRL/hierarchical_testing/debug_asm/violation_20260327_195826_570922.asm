.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -82 # instrumentation
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax 
mov rax, rdx 
jnb .bb_0.1 
jmp .exit_0 
.bb_0.1:
xor rbx, rdi 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
cmp rax, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
