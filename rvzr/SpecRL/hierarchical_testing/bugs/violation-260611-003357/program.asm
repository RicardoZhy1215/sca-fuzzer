.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi]
movq xmm0, rdx
movd xmm6, edx
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx]
movd esi, xmm7
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx]
sbb rcx, rbx
sbb rdx, rbx
and rdx, 0b1111111110000 # instrumentation
movups xmm0, xmmword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rsi]
and rdx, 0b1111111110000 # instrumentation
movups xmm6, xmmword ptr [r14 + rdx]
pmuludq xmm6, xmm2
sub rcx, rdi
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdi
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rdi
movd eax, xmm6
sub rdx, rax
movd xmm6, ebx
pxor xmm1, xmm2
and rbx, 0b1111111110000 # instrumentation
movups xmm5, xmmword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rax
sbb rdi, rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
