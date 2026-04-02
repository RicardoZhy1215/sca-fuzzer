.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -6 # instrumentation
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rsi 
mov rdx, rax 
add rcx, rax 
add rdx, rcx 
mov rdx, rsi 
jnl .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rax, rdx 
cmp rax, rbx 
xor rax, rbx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1 
add rsi, rax 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdx 
xor rsi, rdx 
add rsi, rdx 
xor rsi, rdi 
add rcx, rbx 
cmp rcx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
