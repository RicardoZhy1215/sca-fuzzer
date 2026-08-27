.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rsi]
test rsi, rsi
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rsi]
lea rsi, qword ptr [rsi + rsi + 1]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi
lea rsi, qword ptr [rsi + rsi + 1]
and rcx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
