.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rcx, 1392
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi
xor rax, rsi
and rdi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdi]
xor rbx, rsi
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 536
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rbx
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi
mov rcx, rax
lea rbx, qword ptr [rsi + rbx + 1]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
