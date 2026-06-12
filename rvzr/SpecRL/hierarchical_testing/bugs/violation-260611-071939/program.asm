.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdi
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
sbb rcx, rsi
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi]
xor rbx, rbx
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi]
and rdx, 0b1111111110000 # instrumentation
movups xmm5, xmmword ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rbx
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rax]
and rbx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rbx], xmm7
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rbx
and rsi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rsi
xor rdx, rbx
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rsi
pxor xmm3, xmm3
movd edx, xmm0
lea rcx, qword ptr [rax + rcx + 1]
mov rdx, 64
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx]
mov rdx, 2808
movd edx, xmm7
movd edx, xmm2
and rbx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rbx], xmm0
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax]
movq xmm6, rbx
sbb rax, rax
movd xmm7, ecx
mov rax, rdi
and rcx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rcx]
sbb rcx, rdx
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rbx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rbx], xmm7
sub rcx, rax
pxor xmm7, xmm4
xor rbx, rsi
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2280
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rax]
and rcx, 0b1111111110000 # instrumentation
movups xmm5, xmmword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi]
and rbx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rbx], xmm7
and rsi, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rsi]
xor rsi, rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
