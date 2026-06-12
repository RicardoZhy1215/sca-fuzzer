.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rcx
movq xmm7, rdx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax
pmuludq xmm6, xmm3
movq xmm3, rbx
lea rdi, qword ptr [rdi + rdi + 1]
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rax]
mov rsi, rdi
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rsi
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdx]
mov rcx, 80
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdx]
movq xmm2, rcx
lea rdi, qword ptr [rax + rdi + 1]
pmuludq xmm3, xmm2
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi]
paddq xmm2, xmm3
movd edx, xmm5
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax]
pxor xmm4, xmm1
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
sbb rcx, rdx
lea rax, qword ptr [rax + rax + 1]
and rbx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rbx], xmm4
and rdx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rdx, 0b1111111110000 # instrumentation
movdqu xmm4, xmmword ptr [r14 + rdx]
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdi
and rsi, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rsi]
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm0
and rdi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
