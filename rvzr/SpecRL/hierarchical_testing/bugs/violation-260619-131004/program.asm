.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
setl sil
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
pextrq rdx, xmm5, 0
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
dec rsi
pmuludq xmm4, xmm5
mov rsi, 2888
and rax, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rcx]
sbb rsi, rbx
and rsi, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdi]
sbb rax, rax
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx]
lea rsi, qword ptr [rcx + rsi + 1]
jmp .bb_0.1
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rax]
mov rsi, rcx
setl sil
and rdx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rcx]
setb al
lea rsi, qword ptr [rbx + rsi + 1]
setnz sil
and rax, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rax]
psubq xmm2, xmm5
setb cl
or rsi, rbx
cmp rsi, rbx
lea rdi, qword ptr [rax + rdi + 1]
and rax, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rax]
mov rsi, 1184
mov rax, rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
