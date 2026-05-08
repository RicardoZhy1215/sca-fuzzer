.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rcx, rdx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4424
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4664
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6136
xor rcx, rsi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2664
mov rcx, 5184
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rcx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 8088
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi
mov rsi, 1160
mov rbx, rdi
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
mov rsi, 4112
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6688
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
