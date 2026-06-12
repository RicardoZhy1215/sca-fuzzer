.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111110000 # instrumentation
movups xmm5, xmmword ptr [r14 + rcx]
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm6
pmuludq xmm2, xmm2
mov rcx, 1416
movd esi, xmm0
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rax]
sub rbx, rdi
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rax
and rdi, 0b1111111110000 # instrumentation
movdqu xmm5, xmmword ptr [r14 + rdi]
movd esi, xmm7
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rax
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rbx
pmuludq xmm6, xmm5
mov rbx, 4888
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdx
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdi
movd xmm0, eax
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rsi
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rax
sub rax, rbx
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rdi
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6584
sbb rdi, rdx
pmuludq xmm3, xmm5
and rsi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rsi]
and rdi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdi], xmm0
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi
mov rdx, rax
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
