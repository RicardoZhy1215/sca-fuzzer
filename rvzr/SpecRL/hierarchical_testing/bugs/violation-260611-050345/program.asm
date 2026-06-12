.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
sub rsi, rdi
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rsi
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rcx
sbb rdi, rcx
paddq xmm7, xmm0
mov rbx, 2192
sub rcx, rdi
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi]
movd eax, xmm2
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdi
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdi
and rsi, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rsi], xmm3
pmuludq xmm1, xmm7
mov rdx, 256
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi]
paddq xmm7, xmm5
mov rdx, 496
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rax
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
pmuludq xmm6, xmm7
xor rdi, rax
sbb rbx, rbx
paddq xmm0, xmm7
pxor xmm6, xmm4
sub rdi, rdi
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rax
and rdi, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdi], xmm7
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rax
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi]
paddq xmm6, xmm6
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
lea rdx, qword ptr [rdx + rdx + 1]
movq xmm7, rbx
and rdi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdi]
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm5
and rsi, 0b1111111110000 # instrumentation
movups xmm4, xmmword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi]
xor rbx, rdx
sbb rdx, rdi
and rax, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
