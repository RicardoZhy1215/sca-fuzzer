.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rbx 
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi] 
not rax 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rsi 
inc rsi 
and rdx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rdx] 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rsi] 
test rcx, rcx 
not rcx 
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax] 
mov rcx, 5864 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
