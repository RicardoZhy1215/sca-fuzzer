.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 95 # instrumentation
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 6 
mov rsi, rax 
xor rdx, rsi 
add rcx, rsi 
mov rax, rsi 
xor rdx, rsi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3 
add rdi, 3 
add rax, rcx 
jnbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rsi, rax 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 0 
add rax, 6 
add rsi, rbx 
xor rdx, rdi 
cmp rbx, rcx 
mov rsi, rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
