.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -57 # instrumentation
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rsi 
mov rdx, rbx 
add rax, rcx 
mov rbx, rcx 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rsi, rdi 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 4 
add rsi, rdi 
xor rcx, rbx 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
cmp rdi, rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
