.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -104 # instrumentation
lfence
add rbx, rdx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rdi], 7
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rcx], 5
lfence
add rcx, 0
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rdx], 5
lfence
cmp rbx, rdx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rsi], 7
lfence
add rsi, 1
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], 4
lfence
add rdi, 6
lfence
add rcx, rdx
lfence
jno .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
add rdi, 5
lfence
mov rdi, rbx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rbx], 1
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], 3
lfence
add rsi, 2
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], 7
lfence
add rdx, 0
lfence
cmp rsi, rbx
lfence
mov rdx, rbx
lfence
cmp rdi, rsi
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rdx], 4
lfence
and rax, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rax], 3
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rsi], rdx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rbx], rcx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rbx], 5
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rbx], 7
lfence
add rsi, rdi
lfence
add rdx, 5
lfence
add rdi, 7
lfence
add rcx, 3
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
sbb qword ptr [r14 + rcx], 0
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
