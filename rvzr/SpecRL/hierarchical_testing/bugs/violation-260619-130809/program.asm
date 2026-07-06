.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rcx
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdi]
not rsi
and rsi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdi]
and rsi, rdi
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rcx
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdi
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx]
setb sil
and rsi, rax
setb al
cmp rax, rbx
por xmm4, xmm5
and rsi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rsi]
por xmm1, xmm5
setb al
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6528
setnz dl
and rbx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rbx]
not rsi
pxor xmm3, xmm2
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdi
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rbx
jmp .bb_0.1
.bb_0.1:
and rsi, rax
or rdx, rdi
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rax]
movq xmm1, rcx
pextrq rsi, xmm3, 0
and rax, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx
setnz sil
and rcx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rbx
and rdx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3768
lea rax, qword ptr [rdi + rax + 1]
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rax]
setnz al
and rdi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdi]
pxor xmm3, xmm2
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4040
not rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
