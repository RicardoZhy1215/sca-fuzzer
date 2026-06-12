.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pxor xmm1, xmm5
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax
movd edx, xmm2
pxor xmm3, xmm5
and rax, 0b1111111110000 # instrumentation
movups xmm7, xmmword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx]
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm6
movq xmm1, rsi
sub rdx, rax
sbb rbx, rbx
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm3
and rdi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdi], xmm5
mov rdx, 1624
lea rdx, qword ptr [rbx + rdx + 1]
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rsi
lea rdi, qword ptr [rax + rdi + 1]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rbx
and rsi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rsi]
and rbx, 0b1111111110000 # instrumentation
movdqu xmm4, xmmword ptr [r14 + rbx]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rax, 0b1111111110000 # instrumentation
movdqu xmm5, xmmword ptr [r14 + rax]
mov rbx, rdx
lea rbx, qword ptr [rax + rbx + 1]
xor rax, rax
and rbx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rbx], xmm4
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdx
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rax]
sub rdx, rdi
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm2
lea rdi, qword ptr [rdi + rdi + 1]
pmuludq xmm6, xmm3
xor rbx, rdx
paddq xmm4, xmm0
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi
mov rdx, rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
