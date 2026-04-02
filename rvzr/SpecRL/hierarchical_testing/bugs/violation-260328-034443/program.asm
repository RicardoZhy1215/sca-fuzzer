.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -119 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
mov rsi, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
cmp rcx, rdi
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
mov rsi, rdi
mov rsi, rax
mov rsi, rbx
mov rax, rsi
mov rdi, rax
js .bb_0.1
jmp .exit_0
.bb_0.1:
mov rbx, rdx
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx
add rbx, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rsi, rbx
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
mov rax, rsi
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
