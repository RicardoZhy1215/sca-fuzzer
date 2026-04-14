.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -47 # instrumentation
add rcx, rdi 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
jnbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], 4 
add rdi, rax 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
cmp rbx, rdx 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
add rcx, rbx 
cmp rcx, rdi 
cmp rax, rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
