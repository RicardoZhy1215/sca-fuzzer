.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 26 # instrumentation
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 3 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rbx 
mov rcx, 5 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
add rax, rbx 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
add rbx, 6 
sbb rsi, 1 
add rdx, rax 
sbb rdx, rsi 
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rdi] 
mov rax, 5 
add rbx, rdx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
add rax, rbx 
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 1 
add rdx, 1 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
cmp rax, rbx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2 
sbb rbx, 7 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rsi 
jle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rbx 
and rbx, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rbx] 
mov rax, 0 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], 4 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdx 
add rbx, rdx 
mov rdx, 5 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1 
xor rcx, rsi 
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], 3 
sbb rax, rdx 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax 
mov rcx, 5 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
