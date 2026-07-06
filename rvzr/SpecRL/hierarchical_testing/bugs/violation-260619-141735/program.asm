.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
paddq xmm0, xmm7
pxor xmm2, xmm3
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
setb sil
neg rsi
and rax, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rax]
pxor xmm7, xmm5
mov rsi, 88
and rax, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rax]
or rsi, rax
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3440
not rsi
jmp .bb_0.1
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
pmuludq xmm0, xmm7
pextrq rsi, xmm2, 0
setb sil
lea rsi, qword ptr [rdi + rsi + 1]
mov rdi, rbx
setb sil
paddq xmm6, xmm0
setb cl
pcmpeqd xmm1, xmm4
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rbx
setl sil
cmp rsi, rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
