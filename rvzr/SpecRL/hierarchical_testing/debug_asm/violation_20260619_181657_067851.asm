.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rsi] 
setnz cl 
setz bl 
pxor xmm6, xmm3 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax 
lea rsi, qword ptr [rax + rsi + 1] 
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx] 
pxor xmm6, xmm5 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rbx 
setnz sil 
setb dil 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5744 
xor rax, rbx 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx 
dec rbx 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rbx] 
psubq xmm5, xmm5 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx] 
pxor xmm6, xmm6 
setnz dl 
jmp .bb_0.1 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rcx 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rax 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rcx 
sub rsi, rcx 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
pmuludq xmm4, xmm5 
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx] 
cmp rsi, rax 
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx] 
setb sil 
pmuludq xmm4, xmm5 
or rax, rbx 
setb sil 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx 
and rcx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rsi] 
psubq xmm4, xmm6 
dec rdi 
psubq xmm7, xmm3 
pxor xmm3, xmm5 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rbx 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdx 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
pxor xmm7, xmm6 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
