.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 127 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
xor rbx, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rsi, rbx
mov rdi, rbx
mov rsi, rcx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
add rcx, rbx
mov rdx, rdi
xor rsi, rax
xor rsi, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rdx, rbx
mov rdx, rax
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
mov rax, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
add rax, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rbx, rdi
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rdx, rbx
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
