.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -94 # instrumentation
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
add rdi, rax 
and rbx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
add rsi, rax 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx 
mov rdi, 3 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
sbb rcx, rbx 
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
xor rbx, rsi 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 6 
and rcx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rdi, rsi 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rbx 
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
add rdx, rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
