.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pxor xmm4, xmm7
and rsi, rdi
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx
and rdi, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rsi
mov rsi, 4984
neg rdx
xor rbx, rax
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdi]
and rcx, rsi
not rbx
adc rdi, rbx
setnz sil
test rcx, rax
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx
and rdx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdx
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rdi
movq xmm2, rsi
and rdx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
psubq xmm5, xmm2
not rdx
cmp rdx, rdx
pand xmm2, xmm5
setl sil
and rcx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rcx
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rsi
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rsi]
mov rdi, rdx
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rax
mov rax, rcx
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rsi]
adc rcx, rdi
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rdi]
por xmm6, xmm2
pextrq rcx, xmm7, 0
dec rsi
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rdx
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
neg rdi
and rdx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rdx]
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
