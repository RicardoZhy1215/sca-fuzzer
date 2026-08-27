.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rbx] 
test rdi, rdx 
and rsi, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
lea rcx, qword ptr [rbx + rcx + 1] 
and rcx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rcx] 
test rdi, rdi 
and rax, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rax] 
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rsi 
test rsi, rdx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rsi] 
lea rsi, qword ptr [rsi + rsi + 1] 
and rsi, rsi 
mov rax, 5960 
and rax, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rsi 
or rdx, rsi 
sbb rbx, rbx 
and rsi, rdx 
and rax, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
