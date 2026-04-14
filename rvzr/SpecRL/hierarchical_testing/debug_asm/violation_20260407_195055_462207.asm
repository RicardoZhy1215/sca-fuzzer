.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 102 # instrumentation
cmp rdx, rbx 
add rcx, 4 
sbb rax, rsi 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rax 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], 0 
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx] 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
