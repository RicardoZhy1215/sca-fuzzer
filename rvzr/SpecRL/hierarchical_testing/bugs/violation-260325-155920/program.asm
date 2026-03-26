.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -34 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
jnp .bb_0.1
jmp .exit_0
.bb_0.1:
add rdx, rbx
mov rcx, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
