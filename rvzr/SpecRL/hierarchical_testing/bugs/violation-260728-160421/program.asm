.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdx
and rbx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rbx]
pmuludq xmm1, xmm5
and rdx, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rsi
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rbx
cmp rsi, rbx
and rax, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rdx
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rcx
inc rsi
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdx
and rdi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rdi]
neg rbx
dec rdx
and rsi, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rsi]
not rax
setb al
not rsi
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rax
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rcx
pxor xmm4, xmm6
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdi
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rsi
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rcx
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
xor rsi, rbx
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdi
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rsi]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rsi
neg rsi
pextrq rax, xmm2, 0
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rdx
and rdx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rdx]
psubq xmm3, xmm0
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rax
paddq xmm7, xmm6
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
