.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3128
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5072
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7392
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3912
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2432
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5728
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 6680
and rdx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3232
dec rsi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 8048
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7352
mov rbx, 7368
sbb rdi, rdx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5728
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
setb dl
mov rdi, 5928
pextrq rdi, xmm5, 0
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rbx]
jmp .bb_0.1
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6968
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7776
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2992
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3336
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1760
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1760
pextrq rsi, xmm6, 0
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdx
and rdx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rbx
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rcx]
setnz cl
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
