.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
setnz sil
not rdi
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
lea rbx, qword ptr [rbx + rbx + 1]
setl al
and rsi, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx
pextrq rdx, xmm5, 0
and rsi, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rsi]
sbb rdx, rsi
cmp rax, rcx
and rsi, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rbx
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rax
and rbx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rbx]
cmp rdx, rdi
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdi
jmp .bb_0.1
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdi]
xor rsi, rsi
and rsi, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
cmp rsi, rsi
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rcx
cmp rcx, rdx
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rax
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx
psubq xmm5, xmm2
setnz dil
dec rax
and rsi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rsi]
setnz bl
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rdx
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rbx
and rsi, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rdi
pxor xmm2, xmm7
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
