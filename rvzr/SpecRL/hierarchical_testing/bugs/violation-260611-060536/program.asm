.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
movd xmm6, esi
movq xmm4, rax
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rcx, 0b1111111110000 # instrumentation
movdqu xmm3, xmmword ptr [r14 + rcx]
pmuludq xmm6, xmm0
sub rsi, rsi
pmuludq xmm4, xmm2
and rbx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rbx]
and rcx, 0b1111111110000 # instrumentation
movups xmm5, xmmword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax]
mov rcx, rax
sub rdi, rax
pmuludq xmm7, xmm2
movq xmm3, rcx
movq xmm7, rdi
and rdi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rdi]
movd xmm4, ebx
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi]
mov rdx, rsi
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rbx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rbx], xmm2
mov rbx, 7552
mov rsi, 5288
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rcx]
pxor xmm4, xmm0
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rcx
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rdi
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rsi
and rbx, 0b1111111110000 # instrumentation
movdqu xmm2, xmmword ptr [r14 + rbx]
xor rcx, rdx
movd xmm3, esi
pxor xmm6, xmm0
pmuludq xmm6, xmm5
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax]
and rbx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rbx], xmm5
paddq xmm7, xmm6
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi]
movd esi, xmm5
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
