.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 34 # instrumentation
mov rcx, rdi
mov rbx, rdi
mov rsi, rax
mov rax, rdi
add rsi, rbx
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
add rbx, rdi
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rcx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
jp .bb_0.1
jmp .exit_0
.bb_0.1:
cmp rax, rsi
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rcx
xor rsi, rcx
cmp rax, rcx
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
