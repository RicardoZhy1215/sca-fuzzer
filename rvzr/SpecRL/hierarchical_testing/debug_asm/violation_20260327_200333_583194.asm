.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -71 # instrumentation
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rcx 
add rdi, rax 
xor rbx, rbx 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
add rcx, rdi 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rsi 
mov rax, rsi 
add rdi, rdx 
add rcx, rbx 
add rdi, rcx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
