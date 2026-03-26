.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -45 # instrumentation
xor rax, rdx
add rsi, rbx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
add rcx, rsi
add rdi, rax
add rdx, rbx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
add rdi, rbx
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 2
add rcx, rax
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
add rsi, rax
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx
mov rbx, rdi
cmp rcx, rsi
cmp rdx, rdi
cmp rdx, rdi
xor rbx, rbx
add rdi, 0
add rsi, rcx
xor rdx, rsi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
jns .bb_0.1
jmp .exit_0
.bb_0.1:
mov rax, rbx
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rax
mov rdx, rcx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
xor rdi, rax
mov rdx, rax
add rax, 3
add rcx, rsi
add rbx, rax
add rdi, rax
mov rax, rcx
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
cmp rax, rbx
add rdx, rdi
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
xor rbx, rdi
cmp rsi, rcx
cmp rax, rcx
cmp rdx, rbx
cmp rbx, rax
mov rdx, rsi
add rdx, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
