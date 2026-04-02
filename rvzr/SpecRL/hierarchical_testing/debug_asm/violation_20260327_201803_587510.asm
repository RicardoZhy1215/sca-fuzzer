.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -44 # instrumentation
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdi 
cmp rbx, rdi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rsi 
mov rcx, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
