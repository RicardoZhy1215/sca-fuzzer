.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -7 # instrumentation
xor rax, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
xor rax, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
