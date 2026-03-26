.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -124 # instrumentation
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 0
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
cmp rsi, rbx
add rax, rbx
add rdx, rbx
mov rdx, rcx
cmp rdx, rcx
mov rbx, rdi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
cmp rsi, rax
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
cmp rdx, rsi
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rax
add rcx, rdi
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
jp .bb_0.1
jmp .exit_0
.bb_0.1:
add rsi, rcx
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdi
cmp rsi, rbx
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
cmp rax, rdx
mov rdx, rdi
cmp rsi, rbx
cmp rsi, rdi
xor rsi, rdi
cmp rbx, rdx
add rsi, rbx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
cmp rsi, rcx
xor rsi, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
