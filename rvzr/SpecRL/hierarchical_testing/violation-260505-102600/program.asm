.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rcx, rdx
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rbx
xor rdi, rsi
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2216
mov rcx, rdx
mov rax, rsi
xor rdi, rcx
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdx
mov rbx, 3584
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
xor rcx, rcx
xor rdi, rdi
mov rdi, rsi
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rax
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
