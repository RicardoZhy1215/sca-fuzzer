.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pand xmm3, xmm2
lea rdx, qword ptr [rax + rdx + 1]
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rbx]
or rax, rsi
pcmpeqd xmm3, xmm2
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rsi
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx
xor rbx, rdx
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rsi
pextrq rbx, xmm7, 0
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rax
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax]
pand xmm4, xmm4
dec rax
and rcx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rcx]
sub rdx, rdx
and rdi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdi]
setnz dl
inc rdi
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
mov rbx, rcx
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rbx
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi
and rsi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
