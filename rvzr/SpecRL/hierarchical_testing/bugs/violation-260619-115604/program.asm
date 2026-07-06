.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rcx
not rbx
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rbx
and rax, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
xor rsi, rdx
setnz sil
setb al
psubq xmm6, xmm0
and rbx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
mov rcx, rdx
and rbx, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rdx
and rax, rbx
mov rdx, 6000
pand xmm5, xmm6
paddq xmm2, xmm4
cmp rdi, rbx
pmuludq xmm6, xmm4
jmp .bb_0.1
.bb_0.1:
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
pxor xmm4, xmm7
and rdi, rdx
mov rax, 2544
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdi
paddq xmm4, xmm1
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdx
setl al
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx
setl al
setnz al
and rbx, rsi
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
setb cl
and rax, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rax]
pextrq rax, xmm5, 0
setnz sil
or rsi, rbx
setnz al
pextrq rdi, xmm5, 0
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5544
setnz cl
pmuludq xmm1, xmm3
and rdi, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rdi]
pxor xmm4, xmm5
and rdi, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
