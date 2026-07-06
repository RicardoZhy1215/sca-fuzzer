.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pmuludq xmm1, xmm1
and rsi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rsi]
psubq xmm5, xmm3
and rbx, rcx
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rsi
cmp rdx, rbx
and rdx, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rdx]
sbb rsi, rcx
cmp rax, rcx
paddq xmm0, xmm4
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2248
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rcx
and rsi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rdi]
setb al
sbb rcx, rdx
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rax
pextrq rcx, xmm4, 0
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rsi
setz al
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
not rdx
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rax
or rbx, rsi
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rdi
setl cl
xor rdi, rbx
and rbx, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rdx]
jmp .bb_0.1
.bb_0.1:
or rdx, rbx
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rsi
setnz dil
mov rcx, 7440
not rax
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rbx
setz al
sbb rsi, rdx
not rcx
setnz dil
and rdx, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
imul rsi, rax
neg rax
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rcx
sbb rcx, rcx
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
