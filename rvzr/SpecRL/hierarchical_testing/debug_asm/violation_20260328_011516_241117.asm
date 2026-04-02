.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 87 # instrumentation
mov rsi, rax 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 0 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
mov rcx, rbx 
jo .bb_0.1 
jmp .exit_0 
.bb_0.1:
add rsi, rdi 
mov rsi, rdi 
mov rsi, rbx 
mov rdi, rax 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
