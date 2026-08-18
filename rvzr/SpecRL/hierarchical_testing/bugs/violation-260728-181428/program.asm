.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rbx]
setl al
movd xmm3, ebx
and rbx, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
neg rax
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdx
and rdi, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rsi
movd xmm3, edx
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rbx
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdi
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rbx
and rdi, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdi
movq xmm6, rdi
sbb rbx, rdi
not rbx
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rax
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx
and rax, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rax]
pextrq rsi, xmm1, 0
mov rdx, rdi
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
setb dl
por xmm3, xmm1
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
neg rcx
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdi
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rdx, rbx
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
adc rax, rbx
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rcx
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdx
inc rdx
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rcx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
