.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + rax] 
MOV rax, rdx 
and rdx, 0b1111111111111 # instrumentation
SBB qword ptr [R14 + rdx], 35 
and rax, 0b1111111111111 # instrumentation
SBB qword ptr [R14 + rax], 35 
and r14 + rax, 0b1111111111111 # instrumentation
IMUL byte ptr [R14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
