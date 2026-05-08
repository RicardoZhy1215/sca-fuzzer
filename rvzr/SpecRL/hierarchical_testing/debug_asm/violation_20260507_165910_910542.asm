.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rsi, qword ptr [rcx + rsi + 1] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax] 
mov rax, 584 
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6264 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
