.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
MOV rcx, 10 
MOV rcx, r14 
ADD rcx, 5 
AND rcx, 0b1111111111111 # instrumentation
MOV rax, qword ptr [R14 + rcx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
