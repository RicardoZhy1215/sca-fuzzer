.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 124 # instrumentation
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rax
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], 1 # instrumentation
cmp rdx, rbx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
cmp rbx, rdx
xor rbx, rbx
xor rdi, rdx
xor rdx, rdx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
mov rdx, rbx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
xor rdx, rax
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax
mov rsi, rax
mov rsi, rbx
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
xor rcx, rax
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
cmp rbx, rdx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
jnle .bb_0.1
jmp .exit_0
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
cmp rbx, rdx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rax, rdx
cmp rax, rdx
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
add rsi, rbx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
xor rdx, rax
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx
xor rdi, rdi
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi
mov rbx, rax
xor rdx, rdi
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi]
xor rcx, rdx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
xor rdx, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
