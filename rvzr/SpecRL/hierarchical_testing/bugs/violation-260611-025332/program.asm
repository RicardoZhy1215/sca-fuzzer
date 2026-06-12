.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
sub rdx, rbx
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rdi]
movd xmm7, edx
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
and rbx, 0b1111111110000 # instrumentation
movdqu xmm1, xmmword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax]
sbb rbx, rax
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rdi, 0b1111111110000 # instrumentation
movdqu xmm6, xmmword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx]
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm7
and rax, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rax], xmm3
movd edi, xmm6
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
lea rdi, qword ptr [rdi + rdi + 1]
mov rbx, 7912
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4120
mov rdi, 1528
sbb rdx, rdx
sbb rbx, rdx
movq xmm0, rax
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rdi
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rdi
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdx
and rdx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rbx
sbb rdi, rbx
mov rdi, 5352
and rdx, 0b1111111110000 # instrumentation
movups xmm6, xmmword ptr [r14 + rdx]
sbb rdx, rsi
mov rdi, rcx
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rcx
movd eax, xmm7
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdx
mov rcx, 4232
movd ecx, xmm4
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
