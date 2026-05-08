.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rsi, rdi 
lea rsi, qword ptr [rdi + rsi + 1] 
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx] 
lea rsi, qword ptr [rsi + rsi + 1] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
