.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rsi]
adc rdx, rsi
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rdi
lea rbx, qword ptr [rax + rbx + 1]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4200
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7280
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
