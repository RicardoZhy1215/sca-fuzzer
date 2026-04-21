.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
mov rdx, rcx
mov rax, rcx
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 0
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 0
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx
mov rax, rsi
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rax
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rcx
and rbx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rcx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rax
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx
and rsi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 1
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
