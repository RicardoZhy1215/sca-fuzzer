.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
add rax, -110 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
