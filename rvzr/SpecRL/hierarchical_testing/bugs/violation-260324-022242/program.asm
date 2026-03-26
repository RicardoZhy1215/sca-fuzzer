.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -89 # instrumentation
cmp rdi, rcx
xor rdi, rbx
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 0
mov rax, 5
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
jp .bb_0.1
jmp .exit_0
.bb_0.1:
add rbx, 0
mov rcx, 2
add rax, 7
add rbx, 4
add rdi, 2
mov rsi, 2
cmp rdx, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
