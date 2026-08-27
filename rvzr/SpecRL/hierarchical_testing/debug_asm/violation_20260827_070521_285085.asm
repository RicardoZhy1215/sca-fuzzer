.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rsi] 
test rbx, rbx 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rbx 
test rcx, rbx 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rax] 
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx] 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rsi 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5512 
xor rax, rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
