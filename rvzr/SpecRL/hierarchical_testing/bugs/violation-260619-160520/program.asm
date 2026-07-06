.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rax]
and rsi, rdi
setnz sil
lea rsi, qword ptr [rdi + rsi + 1]
movq xmm0, rbx
imul rax, rcx
pmuludq xmm0, xmm5
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx]
paddq xmm6, xmm7
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rbx
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rbx
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rcx]
pextrq rsi, xmm2, 0
and rbx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rbx
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
paddq xmm7, xmm6
setb sil
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdi
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rbx
jmp .bb_0.1
.bb_0.1:
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rcx
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4944
pxor xmm0, xmm5
setnz sil
and rcx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rcx]
pxor xmm5, xmm5
not rax
setnz sil
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rax
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx]
psubq xmm6, xmm4
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx]
or rsi, rbx
adc rsi, rbx
or rdi, rbx
sbb rdi, rbx
pxor xmm4, xmm5
imul rsi, rbx
setnz bl
and rbx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx]
and rsi, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
