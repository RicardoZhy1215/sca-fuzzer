.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 11 # instrumentation
mov rax, rdi 
mov rax, rdx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
jnl .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rsi, rbx 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 3 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rdx, rbx 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 2 
mov rsi, rbx 
mov rsi, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
