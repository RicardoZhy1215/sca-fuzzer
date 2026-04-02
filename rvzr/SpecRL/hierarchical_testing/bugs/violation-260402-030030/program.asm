.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 25 # instrumentation
cmp rbx, rcx
cmp rsi, rdi
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rax, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
cmp rdi, rbx
cmp rcx, rbx
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 2
xor rax, rbx
cmp rcx, rdi
cmp rdi, rcx
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
jnp .bb_0.1
jmp .exit_0
.bb_0.1:
add rax, 5
cmp rsi, rdx
add rcx, rdi
cmp rbx, rcx
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
xor rsi, rcx
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
xor rbx, rcx
mov rdx, rbx
xor rbx, rbx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
xor rcx, rax
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 6
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
