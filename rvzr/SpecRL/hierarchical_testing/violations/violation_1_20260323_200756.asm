.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 65 # instrumentation
cmp rcx, rax 
cmp rax, rdx 
xor rax, rdi 
add rcx, rsi 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
add rax, rdx 
xor rsi, rax 
cmp rcx, rdi 
cmp rsi, rdx 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx 
jnb .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rax, rdi 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
