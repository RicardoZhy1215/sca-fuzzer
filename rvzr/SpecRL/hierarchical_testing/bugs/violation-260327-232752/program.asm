.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 72 # instrumentation
add rdx, rcx
mov rdx, rdi
mov rsi, rbx
cmp rsi, rbx
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
cmp rdi, rdx
xor rsi, rdi
mov rdi, rsi
cmp rdx, rdi
mov rsi, rdx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
xor rsi, rax
xor rax, rdi
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
jnp .bb_0.1
jmp .exit_0
.bb_0.1:
cmp rsi, rbx
cmp rdx, rdi
mov rdx, rbx
cmp rdx, rbx
add rdx, rax
xor rax, rdi
mov rdx, rbx
add rcx, rdi
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
