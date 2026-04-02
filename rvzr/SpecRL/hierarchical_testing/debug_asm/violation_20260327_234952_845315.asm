.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -9 # instrumentation
add rsi, rbx 
add rsi, 6 
xor rdx, rdi 
add rsi, 5 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rdi, rbx 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax 
mov rdx, rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
