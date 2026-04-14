.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 46 # instrumentation
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
add rdx, 1 
and rax, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rax] 
jns .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
add rax, 3 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rsi 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rax 
mov rax, rdi 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
