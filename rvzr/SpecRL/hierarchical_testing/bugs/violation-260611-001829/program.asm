.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rbx
paddq xmm1, xmm2
and rsi, 0b1111111110000 # instrumentation
movdqu xmm6, xmmword ptr [r14 + rsi]
paddq xmm3, xmm2
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2080
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
mov rbx, 7376
and rax, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rax]
and rdi, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdi], xmm5
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rsi
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rsi
and rcx, 0b1111111110000 # instrumentation
movups xmm3, xmmword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rsi
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx]
and rdi, 0b1111111110000 # instrumentation
movups xmm1, xmmword ptr [r14 + rdi]
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
pxor xmm7, xmm4
xor rdi, rdi
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
movq xmm4, rbx
movq xmm2, rdi
sbb rcx, rsi
sub rdi, rcx
lea rdi, qword ptr [rcx + rdi + 1]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
sbb rcx, rdx
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rbx]
mov rbx, 6768
movq xmm5, rdi
and rcx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdx
sbb rdx, rcx
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rax
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rbx
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rsi
and rdi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rdi]
sub rcx, rdx
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
