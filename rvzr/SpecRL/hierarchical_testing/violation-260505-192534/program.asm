.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -81 # instrumentation
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi]
jp .bb_0.1
jmp .exit_0
.bb_0.1:
xor rdi, rdi
mov rsi, rbx
mov rsi, rbx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5472
lea rdi, qword ptr [rbx + rdi + 1]
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
