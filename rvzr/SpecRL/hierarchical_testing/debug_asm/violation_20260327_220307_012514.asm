.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 31 # instrumentation
xor rdi, rdi 
add rax, rsi 
add rbx, rdi 
add rdi, rsi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 0 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
mov rsi, rcx 
cmp rbx, rax 
xor rbx, rdi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1 
xor rcx, rbx 
add rax, rsi 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rbx 
add rax, rcx 
add rax, rdi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rcx 
cmp rax, rdx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 7 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
xor rdi, rcx 
mov rdx, rax 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdi 
mov rax, rsi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
add rsi, rdx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdx 
cmp rcx, rbx 
xor rcx, rcx 
xor rdi, rsi 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdi 
xor rcx, rdi 
xor rdi, rdi 
mov rdx, rax 
add rax, 4 
mov rax, rdx 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
xor rdx, rdx 
add rcx, rdx 
cmp rdx, rcx 
mov rax, rdi 
mov rax, rdi 
mov rdx, rcx 
mov rsi, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
