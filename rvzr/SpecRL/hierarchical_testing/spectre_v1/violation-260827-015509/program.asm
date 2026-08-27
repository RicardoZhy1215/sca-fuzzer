.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rax
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
sbb rsi, rdi
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 848
sbb rcx, rcx
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rcx
lea rdx, qword ptr [rdx + rdx + 1]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
