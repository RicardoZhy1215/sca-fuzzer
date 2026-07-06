.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rdi] 
or rax, rdx 
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rdx 
setb sil 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdi 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rbx, rcx 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdx 
or rsi, rbx 
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx] 
pxor xmm5, xmm7 
lea rsi, qword ptr [rcx + rsi + 1] 
and rdi, rbx 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
pxor xmm4, xmm5 
psubq xmm2, xmm3 
and rbx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rbx] 
pxor xmm2, xmm4 
setz sil 
and rdi, rbx 
jmp .bb_0.1 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx] 
or rsi, rax 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax 
pmuludq xmm3, xmm7 
and rax, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdx 
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx] 
pmuludq xmm1, xmm6 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx 
pcmpeqd xmm7, xmm5 
and rcx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rcx] 
setnz sil 
pxor xmm2, xmm1 
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx] 
or rsi, rdx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 880 
and rcx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdi 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdi 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
