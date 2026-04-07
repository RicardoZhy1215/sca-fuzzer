.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -43 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
cmp rbx, rax 
add rbx, rax 
mov rdx, rcx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
mov rdi, rbx 
jp .bb_0.1 
jmp .exit_0 
.bb_0.1:
add rdx, rdi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
cmp rax, rdx 
cmp rbx, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
