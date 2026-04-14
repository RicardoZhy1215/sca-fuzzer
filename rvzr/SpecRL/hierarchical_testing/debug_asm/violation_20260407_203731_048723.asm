.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 12 # instrumentation
sbb rax, rbx 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdi 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], 1 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
