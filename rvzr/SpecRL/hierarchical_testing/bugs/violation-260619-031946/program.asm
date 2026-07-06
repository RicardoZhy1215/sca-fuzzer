.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx
pextrq rdx, xmm3, 0
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx
setl dil
neg rcx
and rdi, rsi
test rdi, rbx
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rdx
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rdi
setz sil
setl sil
and rdx, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdx
and rsi, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rsi]
neg rsi
sub rax, rcx
and rdi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx]
pmuludq xmm1, xmm5
and rbx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rbx]
mov rbx, rax
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdx
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
pand xmm7, xmm3
and rsi, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rsi]
test rsi, rdi
setnz dl
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi]
mov rsi, rdi
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rbx
and rsi, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
psubq xmm0, xmm5
sbb rcx, rbx
and rdi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rcx
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdx
and rbx, rdx
setnz sil
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx]
setnz dil
adc rcx, rcx
jmp .bb_0.1
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
