.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -76 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5312 
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
lea rbx, qword ptr [rdx + rbx + 1] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
