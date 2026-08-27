.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rax 
setnz cl 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rsi 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
