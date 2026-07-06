.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
setnz dil 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3224 
setb sil 
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi] 
or rsi, rcx 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rcx 
mov rsi, rax 
and rdi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdi] 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rax 
and rax, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rax] 
imul rsi, rdi 
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax] 
lea rsi, qword ptr [rdi + rsi + 1] 
setb dil 
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdx] 
adc rax, rbx 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rax 
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx] 
jmp .bb_0.1 
.bb_0.1:
and rdx, rcx 
psubq xmm1, xmm5 
paddq xmm0, xmm5 
mov rsi, rax 
pmuludq xmm0, xmm7 
and rax, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi] 
setnz sil 
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 896 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdx 
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdi 
setb sil 
setb dl 
setb sil 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax 
setl sil 
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdi 
pmuludq xmm3, xmm6 
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx] 
setb dil 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
