.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 114 # instrumentation
lfence
add rdi, 0
lfence
add rax, rcx
lfence
add rax, 5
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rbx], 4
lfence
add rdi, rdx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rdi], 4
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rcx], 4
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rbx], rcx
lfence
add rdx, 3
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rbx], 4
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], 5
lfence
jns .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
xor rcx, rbx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rsi]
lfence
cmp rbx, rax
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rcx], 2
lfence
xor rdi, rax
lfence
and rax, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rax], 4
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], 5
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rsi]
lfence
add rcx, rdx
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
