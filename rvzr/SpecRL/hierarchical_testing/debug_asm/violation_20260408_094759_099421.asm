.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -83 # instrumentation
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
sbb rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
sbb rax, 1 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
add rdi, rbx 
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
mov rdi, rax 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
add rax, rbx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx 
sbb rax, 0 
and rsi, 0b1111111111111 # instrumentation
sbb rsi, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rcx 
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
sbb rdi, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 5 
jl .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rcx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 1 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
xor rdi, rbx 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 4 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], 4 
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
