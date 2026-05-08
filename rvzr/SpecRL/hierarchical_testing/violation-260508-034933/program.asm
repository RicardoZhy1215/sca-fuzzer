.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
mov rcx, rdx
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5240
lea rsi, qword ptr [rax + rsi + 1]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
