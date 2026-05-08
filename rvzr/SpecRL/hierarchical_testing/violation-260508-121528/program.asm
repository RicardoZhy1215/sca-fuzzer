.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rsi
mov rdx, 6928
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
xor rcx, rdi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5224
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax]
xor rax, rsi
mov rdi, 7672
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 448
mov rcx, rdx
lea rax, qword ptr [rdi + rax + 1]
lea rbx, qword ptr [rax + rbx + 1]
xor rdi, rdi
mov rdx, rbx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi
mov rax, rcx
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rcx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 1392
lea rax, qword ptr [rsi + rax + 1]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
