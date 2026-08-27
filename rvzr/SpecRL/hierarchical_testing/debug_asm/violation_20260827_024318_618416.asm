.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
mov rdi, 7784 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rsi 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
