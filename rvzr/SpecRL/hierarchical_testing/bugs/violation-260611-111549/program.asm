.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rax
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rsi]
pmuludq xmm2, xmm6
movq xmm4, rbx
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rbx
and rbx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rbx]
and rbx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rbx], xmm3
movd xmm3, eax
sbb rbx, rax
and rax, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5616
mov rdx, rsi
and rbx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rbx], xmm5
lea rcx, qword ptr [rcx + rcx + 1]
movd xmm6, ebx
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rbx]
lea rcx, qword ptr [rbx + rcx + 1]
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi]
movq xmm6, rax
pmuludq xmm2, xmm2
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1192
mov rbx, rax
mov rdx, rax
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdi
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 176
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2664
movq xmm7, rax
and rax, 0b1111111110000 # instrumentation
movdqu xmm4, xmmword ptr [r14 + rax]
movd edx, xmm2
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rcx]
movd edx, xmm0
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
