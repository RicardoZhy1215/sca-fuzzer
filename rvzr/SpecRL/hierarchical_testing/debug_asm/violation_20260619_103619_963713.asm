.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
psubq xmm3, xmm4 
paddq xmm2, xmm4 
and rdi, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rdi] 
paddq xmm2, xmm2 
not rdx 
not rdx 
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi] 
movq xmm4, rdx 
pmuludq xmm5, xmm6 
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rbx 
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rcx 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx] 
pmuludq xmm4, xmm6 
setb dl 
and rdi, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 1704 
and rsi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rsi] 
setnz al 
jmp .bb_0.1 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rcx 
pcmpeqd xmm2, xmm4 
lea rdx, qword ptr [rax + rdx + 1] 
por xmm5, xmm7 
pcmpeqd xmm1, xmm5 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
pcmpeqd xmm5, xmm5 
and rdx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax] 
setl al 
setl sil 
pand xmm2, xmm0 
and rdi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdi] 
setl sil 
and rcx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rcx] 
cmp rdx, rcx 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rbx] 
setnz dl 
paddq xmm5, xmm5 
mov rcx, 1456 
and rdi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdi] 
setl al 
setnz al 
por xmm2, xmm1 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
