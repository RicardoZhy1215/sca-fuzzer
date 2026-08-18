.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pextrq rcx, xmm4, 0
paddq xmm1, xmm6
not rcx
mov rsi, rdx
and rbx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rbx]
pmuludq xmm2, xmm5
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rdx]
mov rcx, 2720
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
paddq xmm3, xmm7
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx
and rcx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rcx]
dec rbx
dec rdx
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rcx
sub rbx, rdx
pxor xmm1, xmm2
and rdx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdx]
sub rbx, rsi
adc rbx, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
