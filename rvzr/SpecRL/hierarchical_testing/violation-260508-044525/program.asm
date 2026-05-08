.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rcx, rsi
mov rcx, 1712
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 7288
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi]
lea rsi, qword ptr [rax + rsi + 1]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi
mov rcx, rdi
xor rsi, rdx
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi
lea rdx, qword ptr [rcx + rdx + 1]
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 2392
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7672
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
