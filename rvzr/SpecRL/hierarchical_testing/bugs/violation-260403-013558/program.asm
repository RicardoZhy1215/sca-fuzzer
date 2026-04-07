.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 101 # instrumentation
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
cmp rdx, rbx
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdx
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx
mov rdx, rbx
mov rdx, rsi
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rsi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
add rbx, rsi
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], 1 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
jnp .bb_0.1
jmp .exit_0
.bb_0.1:
cmp rdi, rsi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
add rax, rdi
cmp rdi, rsi
mov rbx, rcx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
xor rdx, rax
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdx
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rsi
add rax, rdi
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 7
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
