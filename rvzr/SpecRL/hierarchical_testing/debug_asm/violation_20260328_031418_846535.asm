.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -101 # instrumentation
mov rsi, rbx 
add rsi, rdx 
add rsi, rax 
xor rdi, rdx 
mov rsi, rdi 
mov rax, rdi 
xor rcx, rdi 
mov rdi, rax 
mov rsi, rbx 
mov rdi, rsi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rdx, rbx 
mov rdi, rsi 
mov rbx, rdx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdi 
xor rax, rdi 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
mov rbx, rcx 
jns .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rax, rdx 
cmp rbx, rax 
mov rbx, rcx 
mov rcx, rbx 
mov rdx, rsi 
mov rsi, rbx 
mov rsi, rcx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
xor rcx, rcx 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rax, rdi 
mov rsi, rdi 
mov rcx, rax 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
xor rsi, rbx 
cmp rdi, rdx 
add rcx, rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
