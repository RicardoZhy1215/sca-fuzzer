.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
adc rsi, rcx 
setz cl 
and rdx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rdx] 
paddq xmm4, xmm5 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rax 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
not rsi 
paddq xmm2, xmm1 
imul rsi, rbx 
setb sil 
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rax 
pmuludq xmm5, xmm0 
and rbx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
or rsi, rbx 
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx] 
jmp .bb_0.1 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rbx 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax 
lea rsi, qword ptr [rbx + rsi + 1] 
and rcx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rbx] 
dec rcx 
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdx] 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rdx 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax 
setnz sil 
pmuludq xmm0, xmm5 
mov rdi, 6840 
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx] 
pmuludq xmm2, xmm5 
dec rbx 
and rcx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rcx] 
pxor xmm1, xmm7 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx] 
setl al 
setb sil 
psubq xmm1, xmm4 
movq xmm4, rax 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rbx 
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rbx] 
setb sil 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
