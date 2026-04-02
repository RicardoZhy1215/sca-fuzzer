.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 62 # instrumentation
mov rbx, rdx 
cmp rax, rcx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
cmp rbx, rcx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rcx 
mov rax, rbx 
jnbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
xor rbx, rcx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
xor rax, rcx 
add rsi, rcx 
cmp rcx, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
