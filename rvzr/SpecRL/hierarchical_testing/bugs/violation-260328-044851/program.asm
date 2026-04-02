.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -12 # instrumentation
mov rsi, rdi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rbx, rdi
mov rax, rdi
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
mov rcx, rdi
jns .bb_0.1
jmp .exit_0
.bb_0.1:
mov rsi, rbx
mov rbx, rax
mov rdx, rdi
mov rsi, rbx
mov rdx, rcx
mov rax, rdi
add rbx, rdi
mov rsi, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
