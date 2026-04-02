.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -60 # instrumentation
mov rdx, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rsi, rax
js .bb_0.1
jmp .exit_0
.bb_0.1:
add rcx, rdx
mov rax, rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
