.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -16 # instrumentation
add rsi, rax
xor rcx, rax
mov rax, rcx
xor rbx, rdx
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 5
xor rbx, rcx
add rcx, rsi
cmp rbx, rax
xor rdx, rcx
mov rdi, rdx
xor rbx, rsi
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
mov rsi, rax
jns .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 5
add rsi, rcx
mov rbx, rsi
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rax
cmp rbx, rsi
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rbx
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
