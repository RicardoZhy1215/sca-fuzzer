.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -60 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
sbb rdi, 5 
and rdx, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rcx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
add rax, 5 
and rdi, 0b1111111111111 # instrumentation
sbb rdi, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rcx 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 1 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 5 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rsi] 
cmp rdi, rsi 
and rcx, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rcx] 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
cmp rdi, rsi 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
cmp rdi, rbx 
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi 
and rdi, 0b1111111111111 # instrumentation
sbb rdi, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rax 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
add rdx, rdi 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rsi] 
mov rdi, 2 
xor rbx, rax 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
mov rbx, rsi 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
