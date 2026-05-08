.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rsi, rdx
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi]
xor rdi, rax
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
mov rsi, rdx
xor rsi, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
