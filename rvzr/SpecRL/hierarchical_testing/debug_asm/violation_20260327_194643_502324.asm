.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -107 # instrumentation
add rcx, rbx 
cmp rsi, rdx 
add rax, rbx 
cmp rax, rsi 
add rsi, rax 
mov rdi, rdx 
add rbx, 6 
mov rax, rdi 
cmp rax, rdx 
add rsi, rbx 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
add rdx, rbx 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdx 
xor rax, rbx 
xor rbx, rdi 
cmp rcx, rbx 
cmp rdx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
