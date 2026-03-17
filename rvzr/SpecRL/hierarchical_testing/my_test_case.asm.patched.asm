.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 114 # instrumentation
add rdi, 0
add rax, rcx
add rax, 5
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 4
add rdi, rdx
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 4
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rcx
add rdx, 3
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 4
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5
jns .bb_0.1
jmp .exit_0
.bb_0.1:
xor rcx, rbx
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
cmp rbx, rax
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 2
xor rdi, rax
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 4
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
add rcx, rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
