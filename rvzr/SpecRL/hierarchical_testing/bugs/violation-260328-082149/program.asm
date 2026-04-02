.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 35 # instrumentation
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx
add rax, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rsi, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
mov rcx, rbx
mov rsi, rbx
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
mov rdx, rax
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
mov rbx, rax
jp .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 7
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rdx, rax
mov rbx, rdi
xor rbx, rdi
add rdi, rbx
mov rdx, rdi
mov rsi, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
