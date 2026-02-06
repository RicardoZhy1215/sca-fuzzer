.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
.line_1:
MOV rcx, 10 
.line_2:
MOV rcx, r14 
.line_3:
ADD rcx, 5 
.line_4:
and rcx, 0b1111111111111 # instrumentation
MOV rax, qword ptr [R14 + rcx] 
.line_5:
.line_6:
.line_7:
.line_8:
.line_9:
.line_10:
.line_11:
.line_12:
.line_13:
.line_14:
.line_15:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
