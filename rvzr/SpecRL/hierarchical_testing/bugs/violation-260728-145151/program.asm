.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
paddq xmm3, xmm0
and rdx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rbx
mov rsi, 4128
and rsi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rax
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
por xmm0, xmm4
paddq xmm1, xmm0
por xmm7, xmm4
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdx]
pextrq rcx, xmm4, 0
movd xmm7, eax
and rbx, rbx
movd xmm0, edi
dec rsi
paddq xmm6, xmm0
movd xmm0, edi
sub rdx, rbx
neg rbx
pcmpeqd xmm1, xmm0
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdx
not rdi
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rsi
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rsi
movq xmm4, rbx
movd xmm6, esi
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi
setz al
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx
cmp rcx, rsi
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rcx
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rcx
and rbx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rcx
inc rbx
mov rsi, rbx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx
imul rbx, rdx
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rcx
adc rsi, rdx
and rdx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rdx]
dec rcx
dec rbx
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rcx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
