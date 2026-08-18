.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rbx
neg rsi
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rdx]
movd xmm6, edi
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
neg rcx
neg rbx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rdx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx
mov rdi, 3584
pextrq rdi, xmm4, 0
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rbx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdx]
setb dl
adc rbx, rdi
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
pextrq rcx, xmm5, 0
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rdx, rdx
xor rax, rbx
and rdi, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
