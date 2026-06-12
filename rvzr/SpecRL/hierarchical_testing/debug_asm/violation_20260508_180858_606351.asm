.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rdi, qword ptr [rsi + rdi + 1] 
lea rdx, qword ptr [rsi + rdx + 1] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi 
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
