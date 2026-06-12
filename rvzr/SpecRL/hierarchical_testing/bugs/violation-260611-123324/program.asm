.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
sub rdx, rcx
and rax, 0b1111111110000 # instrumentation
movups xmm0, xmmword ptr [r14 + rax]
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm4
movd eax, xmm7
mov rdx, rax
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
sbb rbx, rdi
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
sbb rdi, rax
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6880
and rdi, 0b1111111110000 # instrumentation
movups xmm4, xmmword ptr [r14 + rdi]
and rdi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdi], xmm0
sub rsi, rdx
sbb rdx, rbx
sub rdx, rdi
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rdx
mov rdx, rax
mov rdx, rdi
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rsi
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
sbb rcx, rax
sbb rdx, rax
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
pmuludq xmm1, xmm6
mov rdi, 2952
movd ebx, xmm1
and rdx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdx]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rax
movd edx, xmm0
and rdi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdi], xmm3
mov rbx, rcx
mov rbx, rcx
sbb rbx, rax
and rsi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rsi]
and rdx, 0b1111111110000 # instrumentation
movups xmm2, xmmword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdx]
sub rdx, rdi
and rax, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
