.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rcx
xor rbx, rax
mov rdx, rdi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 1528
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5840
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi
xor rax, rbx
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx]
xor rax, rax
mov rax, rbx
mov rcx, rax
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rsi
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi]
xor rcx, rdx
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rcx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdx
xor rdx, rbx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1600
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
xor rcx, rdi
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdx
mov rcx, rax
xor rcx, rsi
xor rcx, rax
lea rax, qword ptr [rax + rax + 1]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
