.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
movd ebx, xmm6
xor rdx, rcx
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdx
xor rdx, rdx
xor rbx, rbx
sbb rbx, rdi
and rax, 0b1111111110000 # instrumentation
movdqu xmm1, xmmword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2640
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rsi
and rax, 0b1111111110000 # instrumentation
movdqu xmm5, xmmword ptr [r14 + rax]
lea rdx, qword ptr [rdx + rdx + 1]
and rdi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdi]
sbb rdi, rcx
and rbx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rbx], xmm6
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rsi]
and rdi, 0b1111111110000 # instrumentation
movdqu xmm2, xmmword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdi
xor rbx, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
