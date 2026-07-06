.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
adc rax, rbx
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdx
sbb rsi, rax
and rsi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rsi]
pextrq rsi, xmm2, 0
and rsi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rsi]
setnz sil
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rdx]
movq xmm2, rdi
setnz sil
por xmm4, xmm3
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
test rsi, rdi
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4536
pand xmm2, xmm6
setl al
and rsi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 1088
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1968
jmp .bb_0.1
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rax]
pmuludq xmm2, xmm7
adc rsi, rcx
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax]
dec rax
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdi
and rsi, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rsi]
or rsi, rbx
setnz sil
and rcx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rcx]
and rsi, rbx
and rcx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx
paddq xmm2, xmm5
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx]
setnz sil
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
pmuludq xmm7, xmm7
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdi]
pxor xmm2, xmm2
and rdi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdi]
pextrq rax, xmm6, 0
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
