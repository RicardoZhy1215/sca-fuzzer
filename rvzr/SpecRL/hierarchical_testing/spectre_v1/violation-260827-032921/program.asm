.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
lea rcx, qword ptr [rdx + rcx + 1]
not rdi
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi
not rsi
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rsi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 7160
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi
mov rax, 5464
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
