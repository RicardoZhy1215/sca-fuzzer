.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 36 # instrumentation
mov rdi, rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
mov rsi, rbx 
cmp rbx, rax 
mov rsi, rax 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rcx, rax 
mov rax, rbx 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx 
jle .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rsi, rcx 
mov rdi, rbx 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdx 
xor rax, rax 
mov rdx, rdi 
cmp rsi, rbx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
mov rdx, rsi 
cmp rdx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
