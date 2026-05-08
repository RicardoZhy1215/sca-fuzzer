.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 20 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi]
mov rbx, 7376
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3400
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rbx
mov rax, rbx
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi
lea rbx, qword ptr [rsi + rbx + 1]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx
mov rbx, 5456
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4976
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax
lea rdx, qword ptr [rdx + rdx + 1]
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi
jnp .bb_0.1
jmp .exit_0
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx
and rdx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx]
lea rdx, qword ptr [rax + rdx + 1]
xor rbx, rax
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx]
lea rcx, qword ptr [rsi + rcx + 1]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3728
mov rax, 5240
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
