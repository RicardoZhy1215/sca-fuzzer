.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
movd xmm7, ebx
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rax
movd edx, xmm2
sbb rdx, rcx
sub rdx, rax
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7560
movd ebx, xmm1
xor rdx, rbx
movq xmm1, rdi
paddq xmm1, xmm0
mov rdx, rsi
and rdi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdi]
movd xmm7, esi
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rbx
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm0
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1504
paddq xmm1, xmm3
pxor xmm1, xmm6
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax
and rdx, 0b1111111110000 # instrumentation
movdqu xmm6, xmmword ptr [r14 + rdx]
and rax, 0b1111111110000 # instrumentation
movdqu xmm3, xmmword ptr [r14 + rax]
mov rdx, 1744
and rsi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rdi
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdi
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 1184
lea rbx, qword ptr [rax + rbx + 1]
and rdx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdx]
sbb rbx, rsi
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
movd xmm4, esi
sbb rax, rsi
movq xmm7, rdx
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
mov rdx, rbx
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6896
mov rdx, rax
pxor xmm0, xmm4
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
