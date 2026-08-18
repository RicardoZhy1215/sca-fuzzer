.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx
pcmpeqd xmm4, xmm1
pextrq rdi, xmm3, 0
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rdx
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi
and rbx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
por xmm1, xmm0
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rbx, rsi
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rbx
and rax, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rax]
movq xmm6, rbx
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rbx
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
or rdi, rdx
neg rcx
setz cl
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7240
mov rbx, rcx
setz cl
setb cl
neg rbx
sub rbx, rdx
adc rbx, rdi
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
not rcx
movq xmm4, rsi
dec rcx
lea rdx, qword ptr [rax + rdx + 1]
sub rcx, rax
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rbx
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdi
pextrq rcx, xmm0, 0
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
