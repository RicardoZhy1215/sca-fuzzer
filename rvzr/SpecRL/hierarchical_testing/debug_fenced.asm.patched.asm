.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
lfence
mov rcx, qword ptr [r14 + rcx]
lfence
mov rdi, rsi
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rdi], rdx
lfence
mov rax, rbx
lfence
add rdi, 1
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rbx], 1
lfence
add rbx, 5
lfence
cmp rbx, rax
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rax]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], rdx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], 5
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov rax, qword ptr [r14 + rsi]
lfence
mov rcx, 1
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rcx], rcx
lfence
xor rsi, rdi
lfence
cmp rcx, rdx
lfence
sbb rax, 7
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rdx]
lfence
xor rcx, rax
lfence
sbb rdi, rax
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rdi], rsi
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rcx], 5
lfence
mov rdi, rbx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rsi], rdi
lfence
mov rdi, rsi
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
