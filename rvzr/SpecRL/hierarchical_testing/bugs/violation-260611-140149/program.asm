.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rdx, rsi
pmuludq xmm7, xmm7
and rsi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdx
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm3
mov rdx, 1840
and rax, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rax]
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm7
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx
movd xmm7, ecx
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rax
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx
mov rdx, rcx
and rdx, 0b1111111110000 # instrumentation
movdqu xmm4, xmmword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
sub rdx, rax
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdx
pxor xmm1, xmm6
lea rdx, qword ptr [rbx + rdx + 1]
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi]
mov rdx, 6424
and rbx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rbx]
mov rbx, rcx
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rax
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rax
movd xmm6, edi
mov rdi, rdx
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rax
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6392
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
pxor xmm5, xmm2
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3968
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
