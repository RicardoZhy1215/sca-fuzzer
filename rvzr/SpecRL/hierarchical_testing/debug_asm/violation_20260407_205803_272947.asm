.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -50 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx] 
add rax, 0 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 5 
mov rax, rbx 
cmp rdx, rcx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5 
cmp rdx, rbx 
cmp rcx, rdx 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi 
mov rsi, rdi 
cmp rax, rdx 
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rbx] 
xor rdx, rbx 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
mov rdi, rsi 
cmp rbx, rax 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], 0 
add rbx, rsi 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdx 
jp .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rcx, 5 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
mov rbx, rdx 
sbb rax, 0 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdx 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rcx 
sbb rdx, rdi 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
sbb rcx, 4 
add rdx, 5 
and rdi, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdx 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rdi 
and rbx, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdi 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 4 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rcx 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6 
add rdx, 1 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
