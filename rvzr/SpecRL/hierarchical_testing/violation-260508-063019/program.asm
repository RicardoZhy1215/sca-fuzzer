.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rax
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6680
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi]
lea rdi, qword ptr [rbx + rdi + 1]
mov rcx, 4048
mov rdx, rdi
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx
lea rcx, qword ptr [rcx + rcx + 1]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
and rdx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx
xor rax, rax
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 8016
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 6792
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rcx
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi]
lea rdi, qword ptr [rax + rdi + 1]
lea rdx, qword ptr [rsi + rdx + 1]
xor rdx, rbx
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4672
lea rsi, qword ptr [rax + rsi + 1]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
