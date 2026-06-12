.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rsi]
sub rdx, rax
movd edx, xmm5
movd xmm1, esi
movd xmm1, esi
pxor xmm4, xmm1
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 2016
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdx
lea rdi, qword ptr [rbx + rdi + 1]
mov rdx, 1488
mov rdx, rbx
mov rdx, rax
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax]
mov rdx, rdi
movd edx, xmm7
mov rbx, rcx
movq xmm7, rsi
mov rdx, rax
and rdx, 0b1111111110000 # instrumentation
movdqu xmm7, xmmword ptr [r14 + rdx]
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm3
pxor xmm2, xmm3
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7728
and rbx, 0b1111111110000 # instrumentation
movups xmm1, xmmword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rbx]
lea rdx, qword ptr [rdx + rdx + 1]
and rdi, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rdi]
movd xmm3, esi
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax]
mov rdi, 8064
and rbx, 0b1111111110000 # instrumentation
movups xmm0, xmmword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdx
and rbx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rsi
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rdx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rbx]
movd ebx, xmm0
sbb rdx, rsi
pxor xmm1, xmm1
and rdi, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rcx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rcx], xmm2
and rdx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdx]
and rcx, 0b1111111110000 # instrumentation
movdqu xmm2, xmmword ptr [r14 + rcx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
