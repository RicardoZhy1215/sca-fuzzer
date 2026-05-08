.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -55 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
lea rax, qword ptr [rsi + rax + 1]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3272
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdx
xor rcx, rbx
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rsi
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi
mov rsi, rbx
mov rdx, rbx
mov rbx, 2152
lea rdi, qword ptr [rdx + rdi + 1]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
js .bb_0.1
jmp .exit_0
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7704
xor rdx, rsi
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5960
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdx
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6176
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
