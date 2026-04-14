.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 64 # instrumentation
mov rax, rsi
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rcx]
add rbx, rax
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
jp .bb_0.1
jmp .exit_0
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], 5
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
