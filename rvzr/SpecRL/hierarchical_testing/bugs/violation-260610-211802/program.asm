.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rsi]
movd xmm6, ecx
pxor xmm6, xmm6
pxor xmm0, xmm5
sub rax, rsi
movd edi, xmm2
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rdi
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
lea rcx, qword ptr [rdx + rcx + 1]
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rcx
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
pmuludq xmm3, xmm5
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx]
movd xmm2, eax
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi
and rbx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rax]
and rdi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdi], xmm6
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rbx
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rcx
mov rcx, 5464
movd xmm1, edx
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
