.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 20 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi] 
mov rbx, 7376 
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi] 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx 
and rdx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx] 
lea rdx, qword ptr [rax + rdx + 1] 
xor rbx, rax 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax 
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
