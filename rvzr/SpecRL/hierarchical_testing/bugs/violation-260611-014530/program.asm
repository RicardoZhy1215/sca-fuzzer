.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax]
and rbx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rbx], xmm0
and rax, 0b1111111110000 # instrumentation
movups xmm4, xmmword ptr [r14 + rax]
lea rdi, qword ptr [rbx + rdi + 1]
and rdi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdi]
sbb rcx, rdx
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rcx]
movd xmm0, edx
movd xmm1, ecx
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rbx
sbb rdx, rdx
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
movq xmm0, rdi
and rdx, 0b1111111110000 # instrumentation
movups xmm6, xmmword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx
movd xmm2, esi
xor rax, rsi
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rdi
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx]
movd ebx, xmm1
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rax
pmuludq xmm7, xmm4
sbb rbx, rax
sbb rbx, rdx
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rdx
and rdx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rcx
sub rax, rdx
movd xmm6, edi
lea rdi, qword ptr [rdi + rdi + 1]
movd xmm4, edx
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx]
movd esi, xmm6
and rdi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rdi]
mov rbx, 6952
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
paddq xmm6, xmm5
pmuludq xmm1, xmm3
movq xmm0, rax
movd ecx, xmm2
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rbx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rbx], xmm5
movq xmm5, rax
pxor xmm6, xmm2
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
lea rcx, qword ptr [rdx + rcx + 1]
sbb rdx, rbx
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
