.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rdi]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rax]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rax]
lfence
jnp .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rdi]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rbx], 1
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
