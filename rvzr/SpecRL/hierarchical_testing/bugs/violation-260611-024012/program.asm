.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rax
movq xmm0, rax
paddq xmm0, xmm7
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7760
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx]
movd xmm6, ecx
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 6240
sbb rax, rsi
movd xmm7, edx
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
sub rdi, rdx
and rdi, 0b1111111110000 # instrumentation
movups xmm7, xmmword ptr [r14 + rdi]
and rdx, 0b1111111110000 # instrumentation
movups xmm3, xmmword ptr [r14 + rdx]
and rbx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rbx], xmm2
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
pxor xmm6, xmm2
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdx
and rcx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdx
and rsi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rsi], xmm1
sbb rax, rcx
pxor xmm6, xmm6
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdx
movd edi, xmm6
and rbx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rbx], xmm5
and rcx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdx]
pmuludq xmm4, xmm0
movd xmm3, esi
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rax
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rax
sbb rax, rax
movq xmm7, rdx
lea rax, qword ptr [rcx + rax + 1]
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rbx
and rdi, 0b1111111110000 # instrumentation
movups xmm4, xmmword ptr [r14 + rdi]
and rbx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rbx], xmm6
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
