.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -4 # instrumentation
mov rdi, rbx
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx
mov rdx, rdi
mov rax, rdi
mov rsi, rbx
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
mov rdx, rsi
jp .bb_0.1
jmp .exit_0
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
add rdi, rbx
mov rdx, rax
mov rbx, rdi
add rcx, rax
mov rsi, rdi
mov rbx, rax
mov rcx, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
