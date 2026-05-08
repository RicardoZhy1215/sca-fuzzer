.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax]
xor rcx, rcx
and rdi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
xor rsi, rsi
xor rbx, rcx
xor rcx, rsi
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 192
xor rcx, rbx
mov rdx, rdi
xor rcx, rax
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
