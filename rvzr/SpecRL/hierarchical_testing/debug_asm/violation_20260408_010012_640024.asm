.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -61 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
sbb rax, 1 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rsi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx 
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rax 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rcx] 
mov rdx, 5 
cmp rdi, rcx 
and rsi, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
