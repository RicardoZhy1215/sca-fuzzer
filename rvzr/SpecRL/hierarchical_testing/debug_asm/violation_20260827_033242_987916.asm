.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
adc rax, rsi 
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rsi] 
test rax, rcx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rdx] 
not rsi 
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rdi 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
