.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rdx, qword ptr [rax + rdx + 1]
sbb rdx, rsi
and rsi, 0b1111111110000 # instrumentation
movups xmm3, xmmword ptr [r14 + rsi]
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm1
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3280
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax]
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
paddq xmm7, xmm0
pmuludq xmm1, xmm6
sbb rbx, rsi
mov rdx, rdi
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rsi
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rax
movd xmm1, eax
mov rdx, 6336
and rbx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
movq xmm7, rsi
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rdx
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
sub rdx, rdx
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
movd ebx, xmm7
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx]
movd edx, xmm6
xor rdx, rsi
mov rdx, rcx
lea rdx, qword ptr [rsi + rdx + 1]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdi
movd xmm1, ecx
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
sub rsi, rax
xor rdx, rsi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5936
and rax, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3568
movd edx, xmm2
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm6
xor rdi, rdx
lea rdi, qword ptr [rbx + rdi + 1]
pxor xmm1, xmm0
and rdx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi]
mov rdi, 6176
and rcx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
