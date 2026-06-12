.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
sub rdx, rdx
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rax
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rdi
and rdi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdi]
and rcx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rcx], xmm3
pxor xmm0, xmm3
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm6
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rdx
pmuludq xmm3, xmm5
pmuludq xmm0, xmm0
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rax
and rdx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4000
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4176
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx
lea rsi, qword ptr [rcx + rsi + 1]
and rsi, 0b1111111110000 # instrumentation
movdqu xmm5, xmmword ptr [r14 + rsi]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rax]
pmuludq xmm6, xmm6
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rsi]
and rdi, 0b1111111110000 # instrumentation
movdqu xmm0, xmmword ptr [r14 + rdi]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
sbb rdi, rsi
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rax
paddq xmm3, xmm1
pmuludq xmm0, xmm7
movd ecx, xmm6
movq xmm1, rdx
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi
mov rbx, rdi
and rcx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rcx]
pmuludq xmm2, xmm6
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdx
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
mov rax, 56
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
