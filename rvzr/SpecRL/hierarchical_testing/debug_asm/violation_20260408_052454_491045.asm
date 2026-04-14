.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -23 # instrumentation
and rdi, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], 5 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
mov rbx, 5 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
mov rdi, rbx 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
cmp rbx, rdi 
and rcx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
mov rbx, rsi 
add rax, rdi 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rsi 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
add rdx, rsi 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdi 
mov rdi, rbx 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
add rbx, rdx 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
mov rdx, rdi 
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 2 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 5 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
