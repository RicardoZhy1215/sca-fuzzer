.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -40 # instrumentation
mov rdi, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rsi, rbx
mov rdx, rbx
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx
mov rax, rbx
mov rsi, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
xor rax, rdx
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
add rdx, rbx
mov rsi, rbx
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
add rbx, rax
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 3
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 6
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
xor rdx, rcx
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rsi, rbx
mov rax, rcx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
mov rdx, rdi
xor rcx, rbx
mov rdi, rax
xor rdi, rdi
xor rbx, rdi
mov rdx, rdi
add rdx, rdi
add rcx, rax
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
mov rsi, rbx
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx
xor rbx, rdi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
