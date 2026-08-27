.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
mov rdi, rsi 
and rcx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rcx] 
setl al 
lea rdi, qword ptr [rdx + rdi + 1] 
test rdx, rbx 
imul rdx, rsi 
and rdx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdx] 
sbb rax, rsi 
test rdx, rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rcx] 
imul rdi, rsi 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rdi 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rsi] 
inc rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
