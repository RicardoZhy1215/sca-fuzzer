.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7368
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7752
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 8168
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5896
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3480
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3776
mov rax, 6968
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5880
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 8168
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3776
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1576
movq xmm2, rbx
mov rsi, 7296
jmp .bb_0.1
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 608
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 1240
mov rbx, 7536
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rcx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1136
and rcx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rcx]
setl al
and rdx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3128
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7384
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2432
pextrq rax, xmm1, 0
pand xmm1, xmm3
and rdx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rdx]
setnz dl
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6360
and rdx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rdx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
