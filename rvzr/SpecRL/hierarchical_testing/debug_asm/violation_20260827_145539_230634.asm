.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
sub rsi, rax 
dec rsi 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
