.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rax]
test rdi, rcx
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdx
neg rsi
neg rbx
pmuludq xmm3, xmm7
por xmm6, xmm2
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
pmuludq xmm7, xmm3
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rcx
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rax
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
mov rbx, rcx
or rcx, rdx
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rdi
paddq xmm4, xmm6
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6056
and rcx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rcx]
paddq xmm6, xmm2
movq xmm5, rdi
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdx
pmuludq xmm3, xmm2
sbb rcx, rax
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi
movq xmm4, rax
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rax
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax]
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx
cmp rdx, rcx
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi
inc rax
setnz cl
xor rdx, rax
and rbx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdx
pextrq rbx, xmm2, 0
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
paddq xmm3, xmm3
pand xmm4, xmm4
pcmpeqd xmm2, xmm2
pand xmm3, xmm3
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdx
and rcx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rcx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
