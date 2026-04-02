.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 98 # instrumentation
cmp rdx, rbx
mov rdi, rbx
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx
mov rdx, rax
mov rdx, rsi
cmp rdi, rdx
xor rdx, rax
mov rdx, rax
add rsi, rbx
mov rbx, rax
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rbx, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rax, rdi
mov rax, rdi
mov rdi, rsi
js .bb_0.1
jmp .exit_0
.bb_0.1:
xor rdx, rdi
mov rsi, rbx
mov rsi, rdi
mov rsi, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rcx, rax
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
mov rax, rdi
mov rdx, rdi
mov rax, rdi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rdx, rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
