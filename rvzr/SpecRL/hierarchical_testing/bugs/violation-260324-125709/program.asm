.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -48 # instrumentation
cmp rax, rcx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
jp .bb_0.1
jmp .exit_0
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
