.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 3 # instrumentation
mov rdx, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
mov rcx, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
jp .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
mov rdx, rdi
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi
mov rcx, rbx
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx
xor rdx, rbx
mov rsi, rbx
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
