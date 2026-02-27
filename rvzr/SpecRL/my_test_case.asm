.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
