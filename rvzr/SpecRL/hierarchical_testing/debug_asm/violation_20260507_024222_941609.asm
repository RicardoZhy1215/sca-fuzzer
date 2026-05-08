.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rdx, rsi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx 
xor rbx, rdi 
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
