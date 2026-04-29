.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], 4432
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rdi], rdi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rsi], rsi
lfence
xor rbx, rax
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rsi]
lfence
mov rsi, 6048
lfence
mov rsi, 848
lfence
xor rdx, rdx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rdi]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
add rdx, qword ptr [r14 + rax]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rax], rbx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
add rax, qword ptr [r14 + rdi]
lfence
mov rcx, 6352
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rbx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rbx], rbx
lfence
lea rdi, qword ptr [rsi + rdi + 1]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
add rcx, qword ptr [r14 + rdi]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rbx]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
add rcx, qword ptr [r14 + rsi]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rcx], rax
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rax], rdx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rsi], rdx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
add rax, qword ptr [r14 + rsi]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rbx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rbx]
lfence
mov rbx, rax
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rcx]
lfence
mov rcx, 2896
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
mov rbx, qword ptr [r14 + rdx]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], rdx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov rbx, qword ptr [r14 + rsi]
lfence
xor rbx, rbx
lfence
mov rbx, 6600
lfence
xor rdx, rax
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
mov rbx, qword ptr [r14 + rcx]
lfence
mov rax, rcx
lfence
mov rdx, 2640
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
add rax, qword ptr [r14 + rsi]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mov rcx, qword ptr [r14 + rax]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
add rsi, qword ptr [r14 + rcx]
lfence
mov rdi, rbx
lfence
mov rdx, rsi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
add rax, qword ptr [r14 + rsi]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov rdx, qword ptr [r14 + rdi]
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdx], rcx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
add rcx, qword ptr [r14 + rbx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
add rdi, qword ptr [r14 + rbx]
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
