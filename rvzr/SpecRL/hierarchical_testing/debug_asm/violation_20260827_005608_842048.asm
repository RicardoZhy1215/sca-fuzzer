.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 1784 
sbb rcx, rdi 
and rdx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rdx] 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
imul rax, rcx 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rdx 
and rsi, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
