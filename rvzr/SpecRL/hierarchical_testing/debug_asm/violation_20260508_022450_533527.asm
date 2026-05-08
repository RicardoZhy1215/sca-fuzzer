.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rax, 6960 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2824 
mov rdx, 1488 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
