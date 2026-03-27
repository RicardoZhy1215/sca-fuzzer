.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -7 # instrumentation
lfence
xor rax, rbx
lfence
jnle .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
xor rax, rbx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rbx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rbx]
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
