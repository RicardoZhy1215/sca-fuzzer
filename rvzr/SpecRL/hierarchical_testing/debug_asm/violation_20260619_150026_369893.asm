.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx 
setb sil 
psubq xmm6, xmm5 
and rax, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rcx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3608 
adc rsi, rbx 
setnz sil 
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx] 
por xmm4, xmm4 
setb sil 
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx] 
psubq xmm2, xmm0 
and rdx, rbx 
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi] 
setb sil 
pmuludq xmm4, xmm5 
mov rdi, 8160 
mov rdx, rcx 
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
psubq xmm2, xmm1 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
setb sil 
setb sil 
setnz bl 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
jmp .bb_0.1 
.bb_0.1:
pand xmm3, xmm6 
psubq xmm4, xmm5 
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax] 
setnz sil 
and rdx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rbx] 
setnz al 
and rbx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rbx] 
dec rbx 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rdi 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rcx 
adc rdx, rdi 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rbx 
lea rsi, qword ptr [rax + rsi + 1] 
psubq xmm0, xmm5 
setb sil 
pand xmm7, xmm6 
lea rdx, qword ptr [rbx + rdx + 1] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
