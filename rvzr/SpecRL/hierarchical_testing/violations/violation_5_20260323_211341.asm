.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 42 # instrumentation
add rbx, rdi 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
mov rdi, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
