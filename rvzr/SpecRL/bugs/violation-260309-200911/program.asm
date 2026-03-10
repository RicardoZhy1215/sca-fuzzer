.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -103 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
jp .bb_0.1
jmp .exit_0
.bb_0.1:
add rax, -110
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
