.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -121 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
mov rax, 3 
mov rdx, 1 
cmp rbx, rcx 
add rdx, 1 
and rcx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], 1 # instrumentation
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rax 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], 3 
and rax, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rax] 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 1 
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rcx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rax 
cmp rax, rdx 
mov rbx, rax 
and rbx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rbx] 
mov rax, rdi 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
cmp rdi, rbx 
xor rax, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
