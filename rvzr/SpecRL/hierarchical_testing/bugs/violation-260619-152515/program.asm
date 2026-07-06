.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rcx, qword ptr [rdx + rcx + 1]
setnz cl
and rcx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdi
lea rsi, qword ptr [rax + rsi + 1]
lea rsi, qword ptr [rax + rsi + 1]
lea rsi, qword ptr [rcx + rsi + 1]
setl sil
pcmpeqd xmm0, xmm5
and rbx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rbx]
lea rax, qword ptr [rax + rax + 1]
pand xmm4, xmm6
pextrq rsi, xmm6, 0
setb sil
paddq xmm6, xmm5
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rbx
and rdi, rbx
pxor xmm0, xmm7
setb sil
lea rsi, qword ptr [rbx + rsi + 1]
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx]
imul rcx, rbx
movq xmm5, rcx
dec rsi
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
test rdi, rbx
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rbx
setb sil
jmp .bb_0.1
.bb_0.1:
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdi
mov rbx, rdi
por xmm1, xmm6
and rdx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rdx]
pxor xmm5, xmm5
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rcx
setb sil
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rax
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rbx
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx]
not rdx
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5888
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx]
movq xmm1, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
