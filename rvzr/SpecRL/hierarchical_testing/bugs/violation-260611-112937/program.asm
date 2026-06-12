.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
sub rdi, rbx
pmuludq xmm4, xmm0
mov rdx, rsi
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx]
movd xmm4, edx
mov rbx, rax
sub rdx, rax
mov rdx, rax
movd xmm6, eax
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
mov rsi, rdx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2232
movq xmm3, rax
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
and rsi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rsi]
movd ebx, xmm3
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6960
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rax, 0b1111111110000 # instrumentation
movups xmm6, xmmword ptr [r14 + rax]
movd xmm6, edx
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm1
and rax, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
mov rdx, rsi
and rbx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rax
movd esi, xmm1
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi]
movd xmm4, edx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7
and rax, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rax], xmm2
and rdi, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi
pmuludq xmm3, xmm0
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rax
mov rsi, 2992
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdx
and rbx, 0b1111111110000 # instrumentation
movups xmm3, xmmword ptr [r14 + rbx]
mov rdx, rax
sbb rdx, rdx
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rax]
mov rsi, 7440
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 6856
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
