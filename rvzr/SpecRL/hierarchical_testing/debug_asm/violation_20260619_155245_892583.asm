.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rcx] 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rax 
pxor xmm5, xmm1 
and rcx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rcx] 
mov rdx, rcx 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax 
lea rdx, qword ptr [rdx + rdx + 1] 
lea rsi, qword ptr [rax + rsi + 1] 
paddq xmm7, xmm5 
and rsi, rbx 
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx] 
test rsi, rax 
setb sil 
mov rdi, 5832 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rsi 
setnz dil 
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
setnz sil 
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx] 
dec rsi 
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdi] 
and rsi, rbx 
jmp .bb_0.1 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3208 
cmp rsi, rdi 
setb sil 
setb sil 
cmp rsi, rax 
pxor xmm1, xmm5 
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx] 
pmuludq xmm2, xmm1 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 1488 
and rbx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx] 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rcx 
and rdi, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rcx 
and rbx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rbx] 
setb dil 
paddq xmm4, xmm5 
setb sil 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rbx 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
