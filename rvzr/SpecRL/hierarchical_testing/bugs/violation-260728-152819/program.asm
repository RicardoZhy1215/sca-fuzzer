.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
dec rcx
and rbx, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rsi]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
neg rax
xor rcx, rcx
and rbx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx
and rdi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rsi]
cmp rax, rbx
setnz bl
imul rbx, rbx
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rdx]
setb bl
and rax, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx
and rsi, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rsi
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rcx
sub rdx, rbx
pextrq rcx, xmm1, 0
xor rsi, rsi
pextrq rsi, xmm7, 0
setb sil
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rsi]
neg rsi
pcmpeqd xmm2, xmm1
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi
and rbx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
