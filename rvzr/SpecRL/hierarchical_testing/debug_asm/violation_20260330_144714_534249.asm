.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -48 # instrumentation
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdx 
cmp rbx, rcx 
xor rdi, rdx 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rax 
jb .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rsi, rdx 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
cmp rax, rdi 
xor rcx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
