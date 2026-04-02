.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 114 # instrumentation
mov rsi, rbx
mov rdi, rsi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rsi, rax
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx
mov rdi, rcx
mov rbx, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
add rdx, rbx
mov rsi, rbx
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
add rsi, rdx
add rcx, rbx
cmp rdx, rcx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rdx, rbx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rbx
cmp rdx, rbx
cmp rdi, rbx
mov rax, rdi
cmp rbx, rdx
add rdx, rdi
mov rdi, rbx
mov rbx, rcx
mov rdi, rdx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
add rdx, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
add rbx, rdi
mov rsi, rdi
cmp rdx, rdi
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rcx
mov rsi, rax
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rdx, rax
add rcx, rdi
mov rsi, rax
mov rsi, rbx
xor rcx, rbx
xor rdi, rdi
cmp rdi, rbx
mov rdx, rbx
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rax
mov rcx, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
