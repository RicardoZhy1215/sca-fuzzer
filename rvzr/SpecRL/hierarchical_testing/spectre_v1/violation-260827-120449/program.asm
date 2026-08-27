.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
lea rcx, qword ptr [rsi + rcx + 1]
setl dil
and rbx, rax
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rax
and rax, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rax]
inc rbx
cmp rcx, rax
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, rdi
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx]
and rcx, rdi
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi
mov rcx, 3160
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
