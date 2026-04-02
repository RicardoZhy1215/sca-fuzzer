.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 66 # instrumentation
mov rax, rdi 
mov rdx, rsi 
mov rax, rbx 
xor rbx, rdi 
mov rcx, rdi 
mov rsi, rbx 
cmp rdx, rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
jl .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rsi, rbx 
mov rsi, rbx 
add rdi, rbx 
mov rdx, rcx 
add rbx, 6 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax 
mov rax, rsi 
add rsi, rdi 
cmp rdx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
