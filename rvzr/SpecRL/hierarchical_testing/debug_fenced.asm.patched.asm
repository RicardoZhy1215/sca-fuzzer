.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -40 # instrumentation
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rcx]
lfence
add rcx, rdi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rsi], rcx
lfence
xor rsi, rdi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rsi]
lfence
cmp rcx, rdx
lfence
mov rsi, rax
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rsi], 5
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rsi], rax
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rcx]
lfence
mov rdi, rax
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rax]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rsi], rbx
lfence
cmp rax, rdi
lfence
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rsi], rcx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rbx], rdx
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rdx], rax
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rsi], rdi
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rbx]
lfence
mov rdx, rcx
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rdx], rax
lfence
add rsi, rdi
lfence
mov rcx, rdi
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
mul dword ptr [r14 + rdx]
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
