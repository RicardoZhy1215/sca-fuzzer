.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 3 # instrumentation
cmp rdx, rbx 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
mov rdx, rsi 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
add rdx, 1 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rsi] 
mov rax, rdi 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
add rdx, rbx 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdx 
add rdi, rbx 
and rdx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 5 
and rcx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi 
mov rdx, 5 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
