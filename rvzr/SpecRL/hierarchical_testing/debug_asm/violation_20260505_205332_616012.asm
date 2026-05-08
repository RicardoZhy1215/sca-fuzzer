.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -67 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx 
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax] 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
