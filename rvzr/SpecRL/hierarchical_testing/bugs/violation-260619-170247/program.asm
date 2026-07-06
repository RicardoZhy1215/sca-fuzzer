.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
neg rcx
pxor xmm5, xmm6
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rdi
setb sil
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax
pand xmm6, xmm5
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rcx]
pxor xmm5, xmm1
and rsi, rdx
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rbx
psubq xmm3, xmm2
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rbx
jmp .bb_0.1
.bb_0.1:
psubq xmm4, xmm7
psubq xmm0, xmm5
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx]
psubq xmm0, xmm3
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
not rsi
cmp rdi, rcx
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rcx]
and rbx, rcx
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rbx
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
setnz dil
pxor xmm4, xmm5
setb sil
xor rsi, rbx
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rdx
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx]
psubq xmm0, xmm3
pextrq rdi, xmm5, 0
or rsi, rbx
and rdi, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
