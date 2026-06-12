.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rax
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 2728
sub rdi, rdx
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx]
lea rcx, qword ptr [rax + rcx + 1]
and rdi, 0b1111111110000 # instrumentation
movdqu xmm5, xmmword ptr [r14 + rdi]
mov rdi, rbx
mov rsi, rdx
pmuludq xmm7, xmm4
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rdi
pxor xmm7, xmm4
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rbx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rbx
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rdx
sbb rdi, rdi
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rdx
and rdi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdi], xmm3
mov rbx, rdx
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rdi
and rbx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rbx
and rdx, 0b1111111110000 # instrumentation
movups xmm7, xmmword ptr [r14 + rdx]
and rbx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rbx], xmm5
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi
and rcx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rcx], xmm4
sub rcx, rdx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 3120
and rsi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rsi]
sub rbx, rcx
paddq xmm7, xmm1
pxor xmm2, xmm7
movd xmm0, ecx
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx
and rcx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax]
and rcx, 0b1111111110000 # instrumentation
movdqu xmm3, xmmword ptr [r14 + rcx]
sub rcx, rdi
and rdx, 0b1111111110000 # instrumentation
movups xmm7, xmmword ptr [r14 + rdx]
and rax, 0b1111111110000 # instrumentation
movdqu xmm3, xmmword ptr [r14 + rax]
xor rcx, rbx
movd xmm7, edx
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2408
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
