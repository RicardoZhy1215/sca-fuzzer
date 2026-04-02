.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -110 # instrumentation
xor rdx, rbx 
mov rsi, rbx 
mov rsi, rbx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
mov rsi, rdi 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rsi, rdi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rdi, rbx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4 
mov rsi, rbx 
add rax, rdi 
mov rcx, rdx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
