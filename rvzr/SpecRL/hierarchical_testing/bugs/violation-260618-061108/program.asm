.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rdx, 1160
and rcx, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rcx]
inc rcx
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rcx]
setl cl
por xmm2, xmm2
inc rdi
pand xmm3, xmm0
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rcx
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rax
not rdx
and rcx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx]
neg rcx
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rsi
pmuludq xmm0, xmm2
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
mov rdx, rbx
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rsi
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rsi]
psubq xmm4, xmm4
setl dil
cmp rdx, rsi
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rcx
pcmpeqd xmm5, xmm6
and rdi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rdi
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx
and rdx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdx]
and rcx, rax
test rcx, rbx
adc rcx, rbx
pextrq rsi, xmm2, 0
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rax
lea rdx, qword ptr [rax + rdx + 1]
and rdi, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rdi
and rdx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
