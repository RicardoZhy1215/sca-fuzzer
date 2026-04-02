.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -93 # instrumentation
add rax, rdi 
mov rcx, rdx 
mov rsi, rcx 
mov rdx, rbx 
add rdx, rbx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 6 
cmp rsi, rdx 
mov rdi, rax 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rcx 
add rax, rdx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1 
cmp rdx, rdi 
add rsi, rdx 
mov rcx, rdx 
xor rcx, rdi 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rcx, rdi 
mov rax, rbx 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 3 
xor rdi, rdi 
cmp rdx, rsi 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rcx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
xor rsi, rdi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdi 
xor rax, rsi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
mov rcx, rdx 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
cmp rdi, rbx 
mov rcx, rsi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
xor rcx, rdi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
mov rbx, rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
