.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rdi
paddq xmm4, xmm1
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
pextrq rsi, xmm5, 0
dec rsi
mov rsi, rbx
and rdi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdi]
sub rax, rcx
and rsi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rsi]
mov rdx, 6976
setb bl
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7824
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rsi
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rdi
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
not rbx
sbb rax, rdi
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdi
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rdi
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
paddq xmm6, xmm1
pxor xmm0, xmm5
sbb rbx, rcx
sub rax, rbx
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rsi
and rdx, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rdx]
pextrq rax, xmm4, 0
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rbx
inc rbx
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
por xmm4, xmm2
sbb rdx, rdi
sub rbx, rdx
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rdx
and rsi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
pxor xmm4, xmm6
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdx
xor rbx, rbx
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rdi
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rsi
sbb rdx, rdx
and rdi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
