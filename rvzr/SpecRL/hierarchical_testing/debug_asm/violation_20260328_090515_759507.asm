.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 52 # instrumentation
mov rdx, rbx 
add rsi, rbx 
mov rdx, rbx 
mov rdx, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
add rcx, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rsi, rax 
add rsi, rax 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 0 
mov rsi, rbx 
mov rsi, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rsi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rbx 
mov rsi, rax 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rax, rdx 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
add rax, rsi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
jle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rsi, rdi 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rax 
mov rbx, rsi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rax, rcx 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rdx, rbx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
