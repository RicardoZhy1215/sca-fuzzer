.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rbx, 7752
and rsi, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rdx]
pextrq rdx, xmm1, 0
and rbx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rdi
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rbx
adc rdx, rbx
imul rdx, rax
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
and rcx, rax
setl sil
setl bl
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdi]
setb sil
psubq xmm2, xmm1
pextrq rcx, xmm6, 0
inc rdi
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx]
setb sil
paddq xmm6, xmm4
setz dl
and rdi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdi]
inc rbx
and rbx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rbx]
pxor xmm6, xmm0
psubq xmm2, xmm2
neg rdx
neg rcx
and rsi, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rsi
neg rsi
adc rsi, rcx
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx]
mov rcx, rsi
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rdx
imul rbx, rcx
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx
sbb rdx, rsi
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdi
inc rbx
movd xmm4, esi
pxor xmm4, xmm1
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
