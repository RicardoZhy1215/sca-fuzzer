.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 118 # instrumentation
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
mov rdx, rdi 
sbb rax, 5 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], 1 # instrumentation
add rbx, 1 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rsi] 
mov rdi, rcx 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
add rbx, 5 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx] 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rdi, rbx 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
mov rcx, 4 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
sbb rdi, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rbx 
and rcx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
cmp rbx, rdi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
