.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 304 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
mov rax, rsi 
mov rdi, 2888 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdi 
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi] 
mov rbx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
