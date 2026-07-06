.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
psubq xmm2, xmm4
and rcx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rcx]
pand xmm2, xmm4
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rax
lea rdi, qword ptr [rcx + rdi + 1]
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rcx
neg rsi
setb cl
pand xmm2, xmm3
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx]
lea rdx, qword ptr [rcx + rdx + 1]
lea rdi, qword ptr [rbx + rdi + 1]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rbx
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdx
pxor xmm4, xmm1
setnz dil
and rbx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
imul rsi, rax
and rcx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rcx]
jmp .bb_0.1
.bb_0.1:
pextrq rsi, xmm5, 0
setb sil
adc rax, rcx
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
setz sil
and rax, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rax]
pxor xmm4, xmm5
pand xmm2, xmm1
psubq xmm6, xmm5
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rcx
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
setb dil
paddq xmm1, xmm1
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
setb sil
paddq xmm4, xmm0
psubq xmm2, xmm5
pxor xmm4, xmm0
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rbx
pand xmm3, xmm3
psubq xmm7, xmm1
setb dil
neg rsi
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
