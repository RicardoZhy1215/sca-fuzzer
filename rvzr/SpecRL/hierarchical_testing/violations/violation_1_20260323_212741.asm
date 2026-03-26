.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -40 # instrumentation
cmp rax, rsi 
add rdx, rax 
cmp rdi, rbx 
cmp rax, rsi 
xor rdi, rsi 
cmp rdi, rcx 
mov rdi, rcx 
cmp rax, rdx 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rbx, rsi 
cmp rax, rdi 
cmp rcx, rax 
cmp rdx, rdi 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rcx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rax 
mov rdi, rcx 
cmp rcx, rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
