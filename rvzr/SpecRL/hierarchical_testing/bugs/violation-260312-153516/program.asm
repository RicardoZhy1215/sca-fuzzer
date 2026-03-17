.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -98 # instrumentation
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 0
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 0
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 7
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 3
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 0
add rdx, 1
cmp rcx, rax
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdi
add rdi, rdx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 0
add rbx, 7
mov rdx, rax
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3
add rbx, 0
cmp rdx, rdi
xor rdi, rdi
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], 0
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 0
add rcx, rdx
cmp rdi, rcx
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
jns .bb_0.1
jmp .exit_0
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 2
add rdi, rdx
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 7
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2
add rax, 1
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], 0
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 2
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
add rbx, rax
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 7
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 1
add rdi, 2
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 0
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3
add rdi, rsi
add rdx, 3
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 0
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
