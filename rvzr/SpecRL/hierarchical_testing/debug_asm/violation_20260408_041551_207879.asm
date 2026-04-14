.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 61 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rsi 
add rdx, rbx 
mov rdi, rax 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
sbb rcx, 1 
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
jl .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
cmp rdi, rdx 
and rdi, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
