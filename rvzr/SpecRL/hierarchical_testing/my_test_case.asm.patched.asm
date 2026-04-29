.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rcx
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
xor rbx, rdi
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax]
xor rsi, rax
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
lea rax, qword ptr [rdx + rax + 1]
xor rsi, rbx
mov rcx, rdx
mov rdx, rax
lea rbx, qword ptr [rsi + rbx + 1]
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax]
xor rdi, rcx
lea rdx, qword ptr [rdi + rdx + 1]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 1552
mov rbx, 3984
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rax
lea rcx, qword ptr [rax + rcx + 1]
mov rax, 7560
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi
lea rax, qword ptr [rsi + rax + 1]
mov rdi, rdx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi
mov rax, rdi
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
