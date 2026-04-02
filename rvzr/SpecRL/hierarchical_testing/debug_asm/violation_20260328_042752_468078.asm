.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -63 # instrumentation
mov rdx, rdi 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx 
add rbx, rdi 
mov rdx, rbx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
jbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rdx, rdi 
mov rsi, rbx 
mov rbx, rdi 
add rsi, rbx 
mov rdi, rsi 
mov rdi, rsi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
add rcx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
