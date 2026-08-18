.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
adc rax, rcx
pmuludq xmm7, xmm3
not rdx
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax
movd xmm2, esi
setl dl
pand xmm1, xmm5
pxor xmm1, xmm4
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx]
neg rax
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rdi
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
pextrq rbx, xmm5, 0
dec rbx
pxor xmm0, xmm2
paddq xmm0, xmm4
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdx
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 1032
neg rbx
sbb rsi, rcx
and rcx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rcx]
dec rdx
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rcx
imul rdx, rcx
pxor xmm3, xmm4
and rsi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rbx
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx]
mov rsi, rbx
por xmm1, xmm4
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdx
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
pextrq rcx, xmm4, 0
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rcx
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
