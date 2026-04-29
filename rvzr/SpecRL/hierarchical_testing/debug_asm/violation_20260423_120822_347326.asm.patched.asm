.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rbx, rax
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4488
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax
lea rcx, qword ptr [rcx + rcx + 1]
mov rax, rcx
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx]
lea rdx, qword ptr [rcx + rdx + 1]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx]
mov rbx, rax
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
xor rdi, rdi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 2752
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdx
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx]
mov rbx, 6152
mov rsi, rdx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax
lea rbx, qword ptr [rdx + rbx + 1]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdx
xor rbx, rbx
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 6856
mov rax, rdi
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
mov rdi, rax
lea rsi, qword ptr [rcx + rsi + 1]
mov rdx, 2144
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5152
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
