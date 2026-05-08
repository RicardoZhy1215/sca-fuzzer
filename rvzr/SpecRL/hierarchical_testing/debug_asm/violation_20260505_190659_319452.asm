.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 75 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
lea rcx, qword ptr [rdi + rcx + 1] 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdi 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi 
jnl .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
