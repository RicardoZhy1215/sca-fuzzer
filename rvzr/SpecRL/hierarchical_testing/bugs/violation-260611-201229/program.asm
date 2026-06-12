.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 336
lea rax, qword ptr [rdx + rax + 1]
pxor xmm1, xmm2
pmuludq xmm7, xmm4
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rcx]
mov rsi, rcx
and rcx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rcx
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rbx
and rbx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rbx], xmm1
lea rsi, qword ptr [rbx + rsi + 1]
and rsi, 0b1111111110000 # instrumentation
movdqu xmm4, xmmword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdi
pmuludq xmm2, xmm5
sbb rdi, rsi
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx]
sbb rdx, rcx
pxor xmm1, xmm0
and rdi, 0b1111111110000 # instrumentation
movdqu xmm4, xmmword ptr [r14 + rdi]
movq xmm0, rcx
and rax, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rdi]
and rcx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rcx], xmm3
sub rdx, rax
and rdi, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdi], xmm2
and rsi, 0b1111111110000 # instrumentation
movdqu xmm3, xmmword ptr [r14 + rsi]
xor rdi, rsi
and rbx, 0b1111111110000 # instrumentation
movdqu xmm6, xmmword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rsi]
and rax, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rax], xmm2
and rcx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rcx], xmm1
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2656
pxor xmm4, xmm1
and rdi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdi], xmm0
xor rdi, rax
and rdx, 0b1111111110000 # instrumentation
movdqu xmm4, xmmword ptr [r14 + rdx]
mov rbx, 1968
and rdx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdx]
xor rax, rdi
mov rdi, rsi
mov rbx, rdx
and rdx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rdx]
and rdi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdi], xmm1
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6848
mov rdx, rax
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
