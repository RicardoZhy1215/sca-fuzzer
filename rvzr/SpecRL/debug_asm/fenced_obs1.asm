.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
.line_1:
MOV rcx, 10 
lfence  
.line_2:
lfence
.line_3:
lfence
.line_4:
lfence
.line_5:
lfence
.line_6:
lfence
.line_7:
lfence
.line_8:
lfence
.line_9:
lfence
.line_10:
lfence
.line_11:
lfence
.line_12:
lfence
.line_13:
lfence
.line_14:
lfence
.line_15:
lfence
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
