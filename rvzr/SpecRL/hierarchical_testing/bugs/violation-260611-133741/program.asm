.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rdx, rcx
movd edx, xmm2
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rbx
and rax, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rax]
mov rdx, 8
and rdx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdx]
xor rbx, rax
and rax, 0b1111111110000 # instrumentation
movdqu xmm7, xmmword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rcx
mov rdx, rdi
movd xmm1, ebx
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm5
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6896
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm0
sub rdx, rdi
pxor xmm3, xmm1
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rax
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi
sbb rdx, rax
mov rdx, 5624
mov rdx, 4392
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rbx
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax]
and rsi, 0b1111111110000 # instrumentation
movdqu xmm7, xmmword ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rbx
pxor xmm3, xmm6
lea rdx, qword ptr [rdx + rdx + 1]
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rbx
mov rdx, rbx
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax]
movq xmm0, rdi
and rdi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdi], xmm1
mov rbx, rax
xor rdi, rax
mov rbx, rdx
and rdx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdx]
movd edx, xmm5
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3992
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm0
mov rbx, rdx
and rdi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx
sbb rcx, rdx
mov rdi, rdx
and rax, 0b1111111110000 # instrumentation
movups xmm4, xmmword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rdi]
movd xmm5, edi
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6208
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
