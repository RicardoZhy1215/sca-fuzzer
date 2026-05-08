.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 1064
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi
and rdx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rsi
mov rsi, 7496
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3448
mov rcx, 5696
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
xor rsi, rsi
and rsi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5200
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
