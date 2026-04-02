.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 74 # instrumentation
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
mov rsi, rbx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
cmp rax, rbx 
add rax, rsi 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
mov rdx, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
mov rax, rsi 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rsi 
mov rax, rbx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rsi, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
