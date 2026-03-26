.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 125 # instrumentation
cmp rcx, rbx 
add rbx, rax 
cmp rsi, rbx 
cmp rcx, rbx 
jo .bb_0.1 
jmp .exit_0 
.bb_0.1:
add rax, rcx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
