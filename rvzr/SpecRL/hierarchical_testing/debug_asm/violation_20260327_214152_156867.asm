.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -64 # instrumentation
cmp rbx, rdi 
mov rax, rdi 
xor rax, rdi 
xor rdi, rdi 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rcx 
jnbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2 
mov rax, rdi 
mov rdi, rsi 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
mov rax, rdx 
xor rcx, rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
