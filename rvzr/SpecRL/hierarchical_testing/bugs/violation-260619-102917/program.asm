.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx]
setl dil
and rax, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rax]
mov rdi, rax
pxor xmm2, xmm3
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi]
setl dl
adc rax, rdi
not rsi
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
lea rax, qword ptr [rbx + rax + 1]
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rcx
setl al
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7960
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
setnz bl
pxor xmm2, xmm2
jmp .bb_0.1
.bb_0.1:
pxor xmm4, xmm2
setnz sil
imul rsi, rbx
and rcx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rcx]
neg rdx
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax]
mov rdx, rdi
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
movq xmm4, rdi
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
cmp rax, rax
pcmpeqd xmm2, xmm6
and rsi, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rsi]
pcmpeqd xmm2, xmm4
paddq xmm4, xmm4
and rsi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rsi]
pcmpeqd xmm5, xmm1
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rbx]
paddq xmm2, xmm4
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rbx
and rax, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
