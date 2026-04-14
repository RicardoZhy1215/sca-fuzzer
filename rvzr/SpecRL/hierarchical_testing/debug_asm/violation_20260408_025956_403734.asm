.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 42 # instrumentation
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
mov rax, rdi 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rdi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 1 
mov rax, 5 
and rdx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rdx] 
add rdx, rsi 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 1 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rax, rsi 
sbb rdx, 5 
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rcx] 
sbb rbx, 5 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
cmp rdi, rbx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdi 
jnbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
xor rdx, rdi 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], 5 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rcx] 
cmp rax, rbx 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rax 
mov rdx, rbx 
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], 3 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
mov rdi, 1 
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 5 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
