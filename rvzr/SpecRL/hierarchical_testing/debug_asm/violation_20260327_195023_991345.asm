.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -35 # instrumentation
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rbx 
mov rbx, rdi 
jl .bb_0.1 
jmp .exit_0 
.bb_0.1:
add rsi, rdi 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
