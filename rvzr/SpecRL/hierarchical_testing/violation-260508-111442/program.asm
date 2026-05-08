.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2296
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx
mov rdx, 5888
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5344
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rcx
xor rdx, rbx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax]
lea rdx, qword ptr [rdx + rdx + 1]
xor rax, rsi
mov rsi, 4816
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 920
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
xor rax, rsi
xor rsi, rcx
xor rcx, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
