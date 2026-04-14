.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 40 # instrumentation
add rcx, rdi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
sbb rdx, 0 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], 4 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdi 
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rax 
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax] 
sbb rax, 4 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
mov rax, rdi 
add rax, 1 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rax 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rbx 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rsi] 
sbb rdx, 1 
cmp rdx, rdi 
add rdx, rcx 
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
sbb rcx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdi 
add rdi, 6 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
mov rdx, 1 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 2 
and rbx, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rbx 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
add rax, 3 
sbb rdx, 5 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rsi 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 1 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
