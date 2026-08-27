.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
mov rbx, 4728 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
imul rsi, rbx 
and rax, rdi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rax 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rsi 
mov rax, 7880 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
