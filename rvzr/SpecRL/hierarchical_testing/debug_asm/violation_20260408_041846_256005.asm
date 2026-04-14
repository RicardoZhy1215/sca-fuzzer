.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 61 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rsi 
add rdx, rbx 
mov rdi, rax 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
sbb rcx, 1 
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], 2 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdi 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
jl .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
cmp rdi, rdx 
and rdi, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rax 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 5 
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx] 
xor rsi, rdi 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 1 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
add rdx, rbx 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rcx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
