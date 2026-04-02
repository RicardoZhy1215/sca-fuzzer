.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -55 # instrumentation
mov rsi, rbx 
cmp rax, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rsi, rbx 
mov rax, rbx 
mov rsi, rdx 
mov rcx, rbx 
mov rdx, rbx 
mov rax, rsi 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
