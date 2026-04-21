.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -13 # instrumentation
sbb rbx, 7
sbb rdi, 7
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdi
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
xor rcx, rdx
and rdx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdx]
mov rdi, 2
sbb rdx, 4
cmp rdi, rcx
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax]
sbb rbx, rdx
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
jp .bb_0.1
jmp .exit_0
.bb_0.1:
cmp rbx, rax
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdi
and rcx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rax
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rcx
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi]
add rcx, 0
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
