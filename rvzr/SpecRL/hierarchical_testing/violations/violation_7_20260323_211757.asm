.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -47 # instrumentation
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
add rcx, 0 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
cmp rax, rdx 
jnbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
cmp rbx, rsi 
cmp rdi, rax 
mov rbx, rcx 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
xor rax, rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
