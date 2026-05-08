.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rcx, rdi
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx
mov rcx, 2352
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi]
mov rax, 7784
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
