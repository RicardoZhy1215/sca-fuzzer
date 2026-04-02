.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -15 # instrumentation
mov rsi, rbx 
add rsi, rbx 
mov rsi, rdi 
mov rdx, rdi 
mov rsi, rax 
add rsi, rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rsi, rbx 
mov rdi, rsi 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rdx, rdi 
mov rsi, rax 
mov rsi, rax 
add rsi, rbx 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rsi, rbx 
mov rdi, rbx 
mov rcx, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
