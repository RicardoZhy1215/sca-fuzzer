.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -113 # instrumentation
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
mov rcx, rbx
mov rsi, rbx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7
cmp rbx, rdi
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
mov rdx, rbx
mov rsi, rax
mov rbx, rsi
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rax, rbx
cmp rbx, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rcx, rsi
mov rsi, rdi
mov rcx, rax
mov rsi, rbx
mov rcx, rax
jnp .bb_0.1
jmp .exit_0
.bb_0.1:
mov rbx, rdi
add rcx, rdx
mov rbx, rax
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx
cmp rdx, rbx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
add rsi, rbx
mov rbx, rdi
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
mov rsi, rbx
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rsi
add rbx, rdi
mov rsi, rbx
mov rsi, rdi
mov rsi, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rdi, rsi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
add rsi, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
