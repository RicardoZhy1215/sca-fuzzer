.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rdx
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax
pcmpeqd xmm4, xmm0
and rcx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rbx]
xor rsi, rax
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rax
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rcx
and rbx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rbx]
cmp rdi, rdi
adc rdx, rcx
por xmm3, xmm5
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi
adc rcx, rcx
and rcx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdx]
xor rcx, rdi
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx
por xmm4, xmm7
mov rsi, rbx
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rax
pextrq rsi, xmm1, 0
sbb rdx, rcx
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rcx
and rcx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rcx]
pcmpeqd xmm5, xmm3
adc rdx, rcx
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rcx]
pcmpeqd xmm3, xmm4
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rcx
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rbx]
pxor xmm6, xmm3
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 896
pextrq rdx, xmm5, 0
pcmpeqd xmm5, xmm7
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rbx
neg rcx
pxor xmm6, xmm3
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
