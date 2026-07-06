.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pextrq rsi, xmm4, 0
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rax
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
setb bl
and rbx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
sbb rdi, rbx
and rax, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rax]
pmuludq xmm2, xmm2
setb sil
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx]
setb cl
setnz dil
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rax
setb al
jmp .bb_0.1
.bb_0.1:
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rbx
pmuludq xmm1, xmm4
pcmpeqd xmm5, xmm6
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
movq xmm5, rax
and rbx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rbx]
imul rdx, rbx
and rbx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx
and rdi, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rbx
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
