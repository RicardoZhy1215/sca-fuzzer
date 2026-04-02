.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 120 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rcx, rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
xor rcx, rdi 
xor rax, rcx 
add rdx, rdi 
add rcx, rdi 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rsi 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rdx, rsi 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
mov rbx, rdi 
mov rax, rdi 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 0 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
cmp rbx, rdx 
mov rcx, rbx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 1 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rcx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
