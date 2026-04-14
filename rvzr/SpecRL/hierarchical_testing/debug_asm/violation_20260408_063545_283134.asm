.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -34 # instrumentation
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
add rdi, rbx 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
jnb .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rax] 
mov rax, 2 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
