.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rsi, rbx
paddq xmm5, xmm1
and rsi, rdi
pxor xmm0, xmm0
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
psubq xmm2, xmm7
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
setb bl
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rcx]
lea rsi, qword ptr [rdi + rsi + 1]
pxor xmm5, xmm2
paddq xmm4, xmm3
pxor xmm2, xmm7
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
or rsi, rax
adc rdi, rsi
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax]
setl sil
por xmm5, xmm5
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx
sbb rdi, rbx
sbb rsi, rbx
setnz cl
pextrq rsi, xmm7, 0
jmp .bb_0.1
.bb_0.1:
psubq xmm3, xmm5
not rsi
lea rsi, qword ptr [rcx + rsi + 1]
setb sil
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rbx
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rbx
setnz dil
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
imul rsi, rcx
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rcx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
lea rdi, qword ptr [rcx + rdi + 1]
lea rsi, qword ptr [rax + rsi + 1]
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
