.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
setnz dil
pxor xmm1, xmm0
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
lea rdi, qword ptr [rsi + rdi + 1]
mov rbx, rdx
and rax, rcx
neg rbx
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rbx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
neg rsi
and rax, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rax]
setb sil
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
imul rdi, rcx
and rax, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rax]
setb bl
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax
pextrq rcx, xmm1, 0
and rdx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdx]
dec rbx
and rcx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
