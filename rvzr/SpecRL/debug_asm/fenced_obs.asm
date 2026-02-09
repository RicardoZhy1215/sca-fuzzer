.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
.line_1:
lfence
AND rcx, 0b1111111111111 # instrumentation
lfence
IMUL byte ptr [R14 + rcx] 
lfence
.line_2:
lfence
ADD rdx, 5 
lfence
.line_3:
lfence
CMP rdx, rdx 
lfence
.line_4:
lfence
AND rax, 0b1111111111111 # instrumentation
lfence
IMUL byte ptr [R14 + rax] 
lfence
.line_5:
lfence
ADD rdx, 5 
lfence
.line_6:
lfence
XOR rdx, rdx 
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
