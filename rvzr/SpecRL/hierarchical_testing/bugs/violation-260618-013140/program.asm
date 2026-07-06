.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rdx]
and rdi, rsi
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rbx]
pextrq rcx, xmm6, 0
not rbx
xor rcx, rax
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi
cmp rdx, rax
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
setnz dil
and rsi, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi
cmp rsi, rcx
pand xmm5, xmm2
and rdx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rbx]
paddq xmm1, xmm4
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rax]
cmp rbx, rdi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rbx
and rsi, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx]
setnz cl
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rsi
and rdx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdx]
lea rax, qword ptr [rdx + rax + 1]
and rsi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
xor rdx, rax
sbb rax, rbx
psubq xmm7, xmm5
pmuludq xmm2, xmm6
neg rdi
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rcx
mov rsi, rdx
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rsi
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rbx
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rdx
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdx
mov rbx, 2312
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
