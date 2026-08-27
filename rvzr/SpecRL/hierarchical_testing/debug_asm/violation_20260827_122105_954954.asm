.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rdi] 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rsi 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi 
and rdx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rbx 
setnz dil 
and rsi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rsi] 
and rcx, rax 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rax] 
sub rdi, rsi 
inc rbx 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rsi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 3744 
and rsi, rdx 
imul rcx, rax 
test rdi, rbx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdi 
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
