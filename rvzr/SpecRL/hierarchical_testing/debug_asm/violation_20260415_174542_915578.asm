.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 4 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdi 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rcx 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], 6 
sbb rdi, rdx 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
add rax, 2 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rax 
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx] 
mov rbx, 0 
sbb rdx, 5 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 6 
cmp rcx, rdi 
and rcx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rbx 
sbb rsi, 4 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rcx 
jp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], 0 
and rax, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdx 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
cmp rax, rbx 
mov rsi, 0 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
add rdx, 6 
add rdx, rdi 
and rdx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rdx] 
add rcx, 0 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 0 
sbb rcx, 3 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdx 
add rcx, 7 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx 
and rcx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 1 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], 0 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rax 
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx] 
mov rdi, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
