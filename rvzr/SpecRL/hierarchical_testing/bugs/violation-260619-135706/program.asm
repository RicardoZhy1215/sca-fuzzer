.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
and rsi, rdx
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx]
paddq xmm5, xmm5
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
pand xmm1, xmm1
and rax, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rax]
setz sil
imul rsi, rbx
adc rax, rcx
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdx
pextrq rcx, xmm7, 0
not rdi
and rax, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
mov rax, rbx
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdi
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
jmp .bb_0.1
.bb_0.1:
test rsi, rdi
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdi]
setb sil
setz cl
lea rsi, qword ptr [rdx + rsi + 1]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax
lea rsi, qword ptr [rdi + rsi + 1]
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rcx
mov rsi, rax
paddq xmm5, xmm5
movq xmm3, rdx
and rdi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdi]
and rsi, rbx
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
setb sil
xor rdi, rbx
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx]
pmuludq xmm2, xmm0
mov rdi, 1384
psubq xmm3, xmm5
pxor xmm0, xmm6
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
