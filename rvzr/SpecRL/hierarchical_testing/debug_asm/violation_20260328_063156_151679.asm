.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 112 # instrumentation
mov rcx, rbx 
mov rax, rsi 
mov rcx, rdi 
add rsi, rbx 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
cmp rdx, rsi 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
mov rsi, rbx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rsi, rbx 
mov rdi, rbx 
mov rdx, rdi 
jp .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rsi, rbx 
mov rcx, rbx 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
mov rax, rcx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
