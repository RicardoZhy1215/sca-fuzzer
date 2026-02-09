.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
.line_1:
AND rcx, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + rcx] 
.line_2:
ADD rdx, 5 
.line_3:
CMP rdx, rdx 
.line_4:
AND rax, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + rax] 
.line_5:
ADD rdx, 5 
.line_6:
XOR rdx, rdx 
.line_7:
XOR rax, rdx 
.line_8:
JMP .line_1 # instrumentation
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
