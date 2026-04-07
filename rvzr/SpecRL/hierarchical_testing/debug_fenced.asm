.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 11 # instrumentation
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rax]
lfence
mov rax, rdi
lfence
xor rax, rax
lfence
jno .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rax]
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
