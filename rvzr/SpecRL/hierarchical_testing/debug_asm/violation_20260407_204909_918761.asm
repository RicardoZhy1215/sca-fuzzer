.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -50 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx] 
add rax, 0 
jp .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rcx, 5 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
