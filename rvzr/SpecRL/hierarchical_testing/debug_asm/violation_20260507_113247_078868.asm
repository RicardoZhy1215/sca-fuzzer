.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
lea rsi, qword ptr [rbx + rsi + 1] 
lea rcx, qword ptr [rcx + rcx + 1] 
lea rsi, qword ptr [rdx + rsi + 1] 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
