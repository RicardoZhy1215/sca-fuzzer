.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
paddq xmm3, xmm5 
pcmpeqd xmm6, xmm4 
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax] 
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx 
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx] 
setnz sil 
and rax, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx] 
lea rsi, qword ptr [rax + rsi + 1] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rdx] 
setnz sil 
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rdi] 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
setb sil 
and rsi, rax 
mov rax, 5408 
jmp .bb_0.1 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rcx 
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx] 
setnz dil 
xor rsi, rax 
dec rsi 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax] 
pcmpeqd xmm7, xmm6 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdi] 
or rsi, rdi 
psubq xmm6, xmm3 
setb sil 
and rbx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rbx] 
paddq xmm1, xmm1 
psubq xmm5, xmm4 
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx] 
xor rsi, rbx 
setl sil 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
cmp rax, rbx 
lea rdi, qword ptr [rbx + rdi + 1] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
