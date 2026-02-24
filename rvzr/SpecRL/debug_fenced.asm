.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
dec di
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
or word ptr [r14 + rdx], 0b1000 # instrumentation
lfence
and byte ptr [r14 + rdx], 0b1111111111000 # instrumentation
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
and rax, 0b1111111111000 # instrumentation
lfence
.bb_0.1:
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
