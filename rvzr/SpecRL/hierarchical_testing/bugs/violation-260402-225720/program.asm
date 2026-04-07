.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 15 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7
xor rdx, rdx
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi]
mov rdx, rdi
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rsi
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
mov rbx, rdi
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx]
xor rsi, rsi
mov rdx, rax
mov rcx, rdi
add rdi, rsi
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rsi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rcx]
add rbx, rax
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
js .bb_0.1
jmp .exit_0
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
add rbx, rsi
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
cmp rbx, rsi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rbx, rsi
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], 1 # instrumentation
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx]
add rdi, rax
cmp rax, rdx
xor rdi, rdi
cmp rax, rsi
mov rsi, rbx
mov rsi, rcx
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
cmp rdx, rbx
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx
xor rdi, rax
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdi
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
