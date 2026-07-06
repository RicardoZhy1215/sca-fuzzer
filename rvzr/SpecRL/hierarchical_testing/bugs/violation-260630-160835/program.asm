.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pextrq rsi, xmm3, 0
sub rdx, rbx
setb cl
sub rcx, rbx
and rcx, rax
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx
test rdx, rdx
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rsi
cmp rdi, rdi
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rax
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rdi
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx]
pand xmm2, xmm4
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx
inc rdi
jmp .bb_0.1
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rcx]
setb bl
psubq xmm0, xmm3
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rax
neg rdx
por xmm5, xmm0
not rax
paddq xmm0, xmm7
por xmm1, xmm2
neg rax
pextrq rcx, xmm1, 0
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rax
and rsi, rbx
and rdx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rsi
cmp rbx, rax
sbb rax, rbx
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rsi
paddq xmm0, xmm7
and rdi, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rdi]
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rdi
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
