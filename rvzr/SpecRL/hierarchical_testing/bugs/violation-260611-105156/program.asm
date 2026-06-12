.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 504
movq xmm5, rax
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rax
and rax, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rax], xmm1
pxor xmm3, xmm5
movq xmm6, rax
and rdi, 0b1111111110000 # instrumentation
movups xmm2, xmmword ptr [r14 + rdi]
movd edx, xmm5
lea rcx, qword ptr [rbx + rcx + 1]
mov rbx, 872
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rax
and rcx, 0b1111111110000 # instrumentation
movdqu xmm6, xmmword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rbx]
sbb rcx, rax
movd edx, xmm1
movq xmm5, rcx
mov rbx, 4656
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rbx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rbx], xmm6
sub rcx, rax
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm6
and rax, 0b1111111110000 # instrumentation
movups xmm3, xmmword ptr [r14 + rax]
sub rcx, rdx
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax]
sbb rdx, rbx
and rbx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx]
movd xmm2, eax
and rsi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rsi], xmm0
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdi
movd xmm7, edx
paddq xmm4, xmm5
and rax, 0b1111111110000 # instrumentation
movups xmm5, xmmword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax]
movd xmm6, edi
mov rdx, 8080
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
