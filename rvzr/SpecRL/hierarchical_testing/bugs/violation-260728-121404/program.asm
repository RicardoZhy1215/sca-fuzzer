.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
adc rdi, rcx
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
pextrq rcx, xmm6, 0
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rax
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rax
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
imul rcx, rax
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax
and rax, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rax]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
pxor xmm4, xmm7
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rcx
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
imul rcx, rdx
neg rdx
neg rbx
not rcx
pextrq rcx, xmm4, 0
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rcx
sbb rdx, rax
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rsi
lea rcx, qword ptr [rbx + rcx + 1]
adc rcx, rdx
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rcx
neg rcx
and rsi, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
