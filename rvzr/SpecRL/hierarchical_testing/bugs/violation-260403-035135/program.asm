.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -80 # instrumentation
cmp rbx, rsi
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
cmp rax, rdi
cmp rbx, rdi
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
xor rdi, rdi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
jnz .bb_0.1
jmp .exit_0
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
mov rbx, rdx
xor rdi, rbx
xor rax, rbx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
add rdx, rdi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
cmp rsi, rax
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx
mov rdi, rdx
cmp rax, rdx
xor rdx, rdx
xor rdx, rsi
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx
mov rax, rdx
cmp rax, rbx
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
