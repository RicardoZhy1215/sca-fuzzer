.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 59 # instrumentation
mov rdx, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rbx, rax
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
jnp .bb_0.1
jmp .exit_0
.bb_0.1:
add rax, rdi
mov rax, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rdx, rdi
add rdx, rdi
mov rsi, rbx
mov rsi, rdi
xor rax, rdi
cmp rbx, rdx
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
