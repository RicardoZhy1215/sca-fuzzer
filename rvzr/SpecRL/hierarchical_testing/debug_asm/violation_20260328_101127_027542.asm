.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 20 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rax, rbx 
mov rcx, rbx 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
