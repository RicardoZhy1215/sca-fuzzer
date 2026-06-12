.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax
and rsi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
mov rbx, rax
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
sub rdx, rax
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4064
movd xmm2, eax
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi]
mov rbx, rsi
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rax
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi
pmuludq xmm3, xmm7
movd xmm7, eax
mov rdx, rax
mov rdx, rdi
and rsi, 0b1111111110000 # instrumentation
movdqu xmm3, xmmword ptr [r14 + rsi]
sbb rsi, rax
and rax, 0b1111111110000 # instrumentation
movups xmm3, xmmword ptr [r14 + rax]
pxor xmm0, xmm7
pmuludq xmm2, xmm0
lea rdx, qword ptr [rax + rdx + 1]
sub rdx, rax
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax]
sbb rbx, rax
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rax
pmuludq xmm2, xmm0
movd xmm4, edi
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdi
movq xmm1, rbx
lea rdi, qword ptr [rax + rdi + 1]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2744
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rax
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6392
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rax
and rdi, 0b1111111110000 # instrumentation
movups xmm3, xmmword ptr [r14 + rdi]
and rax, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rax], xmm5
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rax]
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm0
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax
and rsi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rsi]
sbb rdi, rax
and rdx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
