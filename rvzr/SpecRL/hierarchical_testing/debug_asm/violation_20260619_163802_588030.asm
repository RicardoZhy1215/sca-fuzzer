.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pand xmm1, xmm6 
mov rbx, 2664 
and rcx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rcx] 
setb sil 
and rsi, rax 
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx] 
setb sil 
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdi] 
or rsi, rax 
psubq xmm1, xmm1 
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rax 
and rdi, rax 
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx] 
dec rsi 
and rcx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdi 
setl sil 
and rdi, rbx 
adc rax, rdi 
sub rdi, rbx 
setnz sil 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rbx 
or rsi, rbx 
jmp .bb_0.1 
.bb_0.1:
pand xmm7, xmm0 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdx 
mov rdi, rcx 
setb sil 
and rcx, rsi 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
imul rsi, rcx 
and rbx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rcx 
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rbx 
setb sil 
cmp rax, rax 
setnz dil 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
