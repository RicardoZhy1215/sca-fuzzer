.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
test rbx, rsi
sub rdi, rdx
and rcx, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rcx]
pextrq rsi, xmm6, 0
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi
inc rcx
and rsi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rsi
jmp .bb_0.1
.bb_0.1:
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
pmuludq xmm4, xmm1
lea rsi, qword ptr [rcx + rsi + 1]
imul rdx, rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
