.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -36 # instrumentation
add rax, 5 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
add rax, 1 
add rdx, rbx 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdx 
cmp rdx, rdi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
add rdx, rsi 
xor rax, rsi 
cmp rax, rbx 
mov rsi, rcx 
mov rdi, rbx 
jl .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rax, rdi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rsi] 
add rdx, rax 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
add rbx, rsi 
xor rbx, rax 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx 
cmp rdx, rax 
add rbx, rdx 
cmp rsi, rdx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rcx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
xor rcx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
