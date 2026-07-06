.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
movq xmm7, rsi
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
xor rdx, rdi
and rdi, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1216
mov rdx, 3976
not rax
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rax
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdx
pextrq rax, xmm3, 0
and rdx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
por xmm6, xmm5
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
mov rsi, 376
pextrq rax, xmm0, 0
pcmpeqd xmm5, xmm1
and rdi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rdi]
jmp .bb_0.1
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx]
sbb rax, rdx
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rsi
sbb rax, rdx
setz dl
pcmpeqd xmm0, xmm5
neg rcx
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdx
pextrq rax, xmm4, 0
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
