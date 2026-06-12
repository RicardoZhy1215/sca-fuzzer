.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rcx, 3136 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
lea rsi, qword ptr [rax + rsi + 1] 
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rbx 
lea rax, qword ptr [rsi + rax + 1] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
