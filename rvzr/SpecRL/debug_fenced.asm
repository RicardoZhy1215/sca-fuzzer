.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 114 # instrumentation
lfence
jbe .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rcx]
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
