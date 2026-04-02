.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 31 # instrumentation
xor rdi, rdi 
add rax, rsi 
add rbx, rdi 
add rdi, rsi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 0 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
mov rsi, rcx 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
xor rdi, rcx 
mov rdx, rax 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdi 
mov rax, rsi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
