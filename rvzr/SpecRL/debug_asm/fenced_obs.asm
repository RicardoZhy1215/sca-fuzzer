.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
.line_1:
MOV rcx, 10 
.line_2:
lfence  
.line_3:
MOV rcx, r14 
.line_4:
lfence  
.line_5:
ADD rcx, 5 
.line_6:
lfence  
.line_7:
and rcx, 0b1111111111111 # instrumentation
lfence  
.line_8:
MOV rax, qword ptr [R14 + rcx] 
.line_9:
lfence  
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
