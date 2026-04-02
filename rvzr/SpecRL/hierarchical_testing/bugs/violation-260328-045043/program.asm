.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -117 # instrumentation
add rdx, rax
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
mov rbx, rax
jns .bb_0.1
jmp .exit_0
.bb_0.1:
mov rsi, rbx
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
