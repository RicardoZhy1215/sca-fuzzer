.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -88 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
xor rcx, rdx
add rcx, 1
add rax, rcx
cmp rax, rcx
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 5
xor rcx, rdx
add rdi, 4
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdi
add rbx, rcx
cmp rbx, rdi
cmp rax, rsi
add rbx, rdx
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
jns .bb_0.1
jmp .exit_0
.bb_0.1:
mov rdi, rcx
mov rax, rcx
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx
add rax, rsi
xor rdi, rbx
cmp rsi, rax
xor rax, rcx
add rcx, rdx
mov rax, rdx
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rcx
mov rbx, rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
