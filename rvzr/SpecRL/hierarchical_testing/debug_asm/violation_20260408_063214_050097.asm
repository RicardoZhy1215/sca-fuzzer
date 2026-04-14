.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 121 # instrumentation
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rsi 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx 
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rsi 
mov rdi, rbx 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rax 
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
mov rax, 5 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
jl .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
cmp rdi, rbx 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx] 
xor rsi, rdi 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 5 
and rsi, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 1 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdi 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], 1 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
cmp rdi, rbx 
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
add rbx, 6 
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rdi] 
mov rax, rdi 
sbb rdx, rbx 
mov rdi, 2 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
