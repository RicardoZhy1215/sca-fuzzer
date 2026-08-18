.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx
not rsi
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rsi
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax
movd xmm4, eax
pextrq rdx, xmm5, 0
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rcx
and rsi, rsi
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rax]
setz dl
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rax
or rdi, rax
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdx
sub rbx, rsi
and rdi, rdx
psubq xmm6, xmm0
and rdx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
mov rsi, 7840
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2480
pcmpeqd xmm5, xmm1
and rdi, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rdi]
mov rdi, rdx
setb dil
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
