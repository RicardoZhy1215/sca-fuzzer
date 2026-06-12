.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rax
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdi
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rax
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rax]
movq xmm1, rsi
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm4
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rax
lea rdx, qword ptr [rax + rdx + 1]
xor rdx, rsi
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm0
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rsi, 0b1111111110000 # instrumentation
movdqu xmm1, xmmword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rax
mov rbx, rax
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rax, 0b1111111110000 # instrumentation
movdqu xmm1, xmmword ptr [r14 + rax]
movq xmm5, rcx
mov rbx, rax
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi]
movq xmm1, rdi
movd eax, xmm6
mov rdx, rax
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi
movd ebx, xmm1
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rsi
mov rbx, rdx
or rdx, 1 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx
lea rsi, qword ptr [rax + rsi + 1]
pxor xmm7, xmm1
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2400
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rcx
and rbx, 0b1111111110000 # instrumentation
movdqu xmm0, xmmword ptr [r14 + rbx]
mov rdi, 656
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rsi
lea rdi, qword ptr [rdx + rdi + 1]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi
movq xmm4, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
