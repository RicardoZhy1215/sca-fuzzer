.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -114 # instrumentation
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdi] 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
