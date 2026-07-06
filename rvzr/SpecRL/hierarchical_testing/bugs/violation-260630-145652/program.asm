.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pextrq rdi, xmm4, 0
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
cmp rbx, rbx
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
pmuludq xmm1, xmm3
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rsi]
sbb rax, rsi
setz cl
not rsi
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rsi
sbb rax, rbx
jmp .bb_0.1
.bb_0.1:
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rcx
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rbx
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rcx]
imul rbx, rsi
psubq xmm6, xmm2
sub rcx, rbx
dec rdx
adc rax, rsi
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rcx
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdx
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
