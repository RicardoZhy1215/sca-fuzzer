.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -4 # instrumentation
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 5
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rcx
xor rsi, rdi
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi
xor rbx, rdi
add rcx, rax
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
xor rsi, rbx
xor rcx, rdx
mov rax, rsi
add rsi, rdi
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
jp .bb_0.1
jmp .exit_0
.bb_0.1:
mov rdx, rbx
xor rdx, rcx
xor rax, rbx
add rdi, rdx
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
cmp rcx, rbx
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rax
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 0
xor rcx, rdi
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rsi
xor rax, rsi
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
xor rdi, rax
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
xor rcx, rdx
cmp rcx, rsi
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
