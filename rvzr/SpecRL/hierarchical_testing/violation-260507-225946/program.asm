.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
xor rsi, rcx
and rcx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rcx]
lea rsi, qword ptr [rcx + rsi + 1]
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rax
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx
mov rsi, rax
xor rsi, rsi
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rax
mov rbx, 4328
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi
xor rdi, rsi
mov rdx, rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
