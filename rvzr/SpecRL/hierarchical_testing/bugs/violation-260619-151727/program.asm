.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
psubq xmm4, xmm5
and rcx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rcx
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx]
cmp rsi, rax
and rcx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rcx
lea rsi, qword ptr [rdi + rsi + 1]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 288
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4968
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rax
setb sil
mov rdi, 6992
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rdi]
psubq xmm1, xmm5
xor rsi, rbx
and rdx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rbx
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx]
pcmpeqd xmm0, xmm2
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx
jmp .bb_0.1
.bb_0.1:
pextrq rdi, xmm6, 0
pxor xmm4, xmm5
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax]
psubq xmm5, xmm1
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4056
adc rsi, rdx
adc rsi, rax
and rcx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rcx]
setb sil
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rax
and rdx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdx]
and rdi, rcx
mov rsi, 4328
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rbx
setb sil
lea rsi, qword ptr [rax + rsi + 1]
setz cl
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
pxor xmm1, xmm1
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
