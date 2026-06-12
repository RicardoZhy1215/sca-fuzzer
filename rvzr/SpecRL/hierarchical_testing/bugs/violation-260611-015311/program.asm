.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rcx]
sbb rcx, rdx
mov rax, rdx
sub rdi, rdi
xor rbx, rbx
and rbx, 0b1111111110000 # instrumentation
movdqu xmm7, xmmword ptr [r14 + rbx]
movq xmm1, rdi
and rax, 0b1111111110000 # instrumentation
movdqu xmm2, xmmword ptr [r14 + rax]
xor rdx, rdi
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm0
movd xmm1, esi
mov rax, rcx
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rsi
and rdi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rdi]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi
and rdx, 0b1111111110000 # instrumentation
movdqu xmm1, xmmword ptr [r14 + rdx]
and rdx, 0b1111111110000 # instrumentation
movdqu xmm5, xmmword ptr [r14 + rdx]
xor rax, rdx
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rsi]
movq xmm5, rdi
and rbx, 0b1111111110000 # instrumentation
movdqu xmm3, xmmword ptr [r14 + rbx]
xor rcx, rdx
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rcx
lea rsi, qword ptr [rsi + rsi + 1]
and rbx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi]
movd edx, xmm7
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rax, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rax], xmm6
lea rcx, qword ptr [rax + rcx + 1]
and rbx, 0b1111111110000 # instrumentation
movups xmm0, xmmword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rax]
movd xmm2, edi
or rdx, 1 # instrumentation
pmuludq xmm7, xmm3
movq xmm1, rdi
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax
and rsi, 0b1111111110000 # instrumentation
movups xmm5, xmmword ptr [r14 + rsi]
mov rcx, rsi
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
