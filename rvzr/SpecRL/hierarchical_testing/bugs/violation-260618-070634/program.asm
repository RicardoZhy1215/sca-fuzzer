.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdx]
pand xmm3, xmm3
mov rdx, rsi
pextrq rcx, xmm3, 0
inc rdx
and rsi, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rsi]
and rbx, rax
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdi
pand xmm1, xmm7
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rcx]
mov rcx, 3176
pcmpeqd xmm7, xmm3
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx]
xor rsi, rdi
lea rdi, qword ptr [rdx + rdi + 1]
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rax
and rdi, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rcx]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rsi, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi
pmuludq xmm7, xmm5
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
