.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
pmuludq xmm6, xmm4
not rax
pextrq rcx, xmm6, 0
movq xmm2, rdi
and rsi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rcx
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 736
or rbx, rcx
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rsi
or rdx, rbx
xor rsi, rcx
and rcx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
setnz bl
mov rbx, 7456
and rcx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rcx]
adc rdi, rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
