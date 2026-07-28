.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 74 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
jp .bb_0.1
jmp .exit_0
.bb_0.1:
cmp rbx, rcx
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
