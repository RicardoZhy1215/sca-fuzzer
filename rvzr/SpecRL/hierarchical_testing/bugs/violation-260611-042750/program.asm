.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111110000 # instrumentation
movdqu xmm2, xmmword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rsi
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rsi
movd xmm6, edx
mov rcx, 2416
xor rdx, rbx
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rdi
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi]
movd xmm6, eax
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax
pmuludq xmm5, xmm5
sbb rbx, rbx
xor rdi, rdi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi]
movd xmm0, edx
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6920
and rdx, 0b1111111110000 # instrumentation
movups xmm6, xmmword ptr [r14 + rdx]
and rsi, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rsi], xmm0
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdi
and rsi, 0b1111111110000 # instrumentation
movups xmm7, xmmword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi]
mov rdx, rdi
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rcx
and rcx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rcx], xmm3
sub rdx, rsi
mov rax, rdi
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rbx]
pmuludq xmm5, xmm0
and rsi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rsi], xmm5
sub rdi, rdi
and rbx, 0b1111111110000 # instrumentation
movups xmm1, xmmword ptr [r14 + rbx]
paddq xmm4, xmm5
paddq xmm1, xmm0
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6040
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rax]
pmuludq xmm5, xmm4
and rax, 0b1111111110000 # instrumentation
movups xmm4, xmmword ptr [r14 + rax]
sbb rax, rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
