.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pxor xmm4, xmm0
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi
pextrq rbx, xmm2, 0
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rsi
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
jmp .bb_0.1
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
