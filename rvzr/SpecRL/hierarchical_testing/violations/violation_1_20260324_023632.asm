.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 71 # instrumentation
mov rcx, 5 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 6 
jns .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rcx, 0 
xor rbx, rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 6 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
