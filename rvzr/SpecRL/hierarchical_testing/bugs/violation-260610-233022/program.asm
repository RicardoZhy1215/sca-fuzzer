.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7024
pxor xmm4, xmm5
pmuludq xmm3, xmm6
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax
mov rsi, 640
pxor xmm3, xmm0
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdx
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
movd ecx, xmm6
lea rax, qword ptr [rbx + rax + 1]
movd xmm6, esi
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rsi
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rcx]
sub rdx, rax
sub rdi, rsi
and rsi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rsi]
movq xmm6, rcx
and rdx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdx]
movd xmm6, esi
xor rsi, rdx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7280
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm0
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx]
mov rbx, rdx
movd esi, xmm2
and rax, 0b1111111110000 # instrumentation
movdqu xmm5, xmmword ptr [r14 + rax]
mov rcx, 7624
sub rcx, rsi
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3856
and rsi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rcx, 0b1111111110000 # instrumentation
movdqu xmm3, xmmword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdx
mov rax, rdx
mov rsi, 1536
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
