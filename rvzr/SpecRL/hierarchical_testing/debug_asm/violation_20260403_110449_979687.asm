.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 14 # instrumentation
xor rbx, rdx 
cmp rdi, rdx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rsi, rax 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
cmp rax, rbx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
add rax, rdx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
