.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 4 # instrumentation
add rdi, rbx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 3
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7
mov rbx, rdx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4
mov rbx, rax
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rbx
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 7
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 2
xor rsi, rbx
add rcx, 4
cmp rdx, rax
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
jnle .bb_0.1
jmp .exit_0
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rbx
xor rax, rax
mov rcx, rsi
xor rcx, rdi
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], 6
add rsi, 6
add rdi, rcx
add rdi, 3
cmp rsi, rax
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 4
cmp rbx, rsi
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 0
cmp rdx, rbx
mov rsi, rax
mov rsi, rdi
add rdi, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], 4
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 2
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
