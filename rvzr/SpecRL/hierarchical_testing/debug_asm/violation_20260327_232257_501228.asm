.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -37 # instrumentation
mov rdi, rbx 
mov rsi, rbx 
jle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
