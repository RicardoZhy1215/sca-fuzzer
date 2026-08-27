.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi
and rdx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
setnz al
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rcx
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdx
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx
lea rax, qword ptr [rbx + rax + 1]
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
or rax, rax
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rdi
and rcx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rcx]
mov rcx, 256
setnz cl
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rbx
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
