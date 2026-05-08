.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3400
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5512
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 856
xor rdx, rbx
mov rdx, rdi
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6592
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
