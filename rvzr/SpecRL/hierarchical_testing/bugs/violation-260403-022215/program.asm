.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -117 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rcx
cmp rdx, rbx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
cmp rdx, rcx
xor rbx, rax
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
cmp rcx, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi
xor rdx, rsi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rdx, rsi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
jnle .bb_0.1
jmp .exit_0
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rax
add rsi, rax
mov rdx, rbx
mov rax, rdx
xor rdx, rcx
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rsi
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
cmp rdx, rbx
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
cmp rdi, rbx
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
cmp rbx, rax
cmp rdx, rdi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
cmp rcx, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
cmp rdi, rdx
xor rdi, rbx
xor rdx, rsi
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
