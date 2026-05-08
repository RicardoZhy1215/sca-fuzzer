.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx
mov rax, rsi
xor rbx, rcx
xor rbx, rdi
mov rdi, 592
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rbx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rcx
mov rdx, rsi
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
xor rsi, rcx
lea rsi, qword ptr [rsi + rsi + 1]
mov rdi, rdx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1312
mov rax, 832
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdi
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rcx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 344
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rbx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rsi
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rbx
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx
lea rcx, qword ptr [rdx + rcx + 1]
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
lea rdi, qword ptr [rax + rdi + 1]
xor rdx, rdi
xor rdi, rax
and rdi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
