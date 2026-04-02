.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 111 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6 
mov rsi, rax 
add rsi, rbx 
mov rdx, rax 
xor rbx, rdi 
xor rdi, rdx 
xor rsi, rax 
xor rcx, rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
xor rcx, rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
xor rsi, rdi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
mov rdx, rcx 
mov rdi, rbx 
mov rdx, rcx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rcx 
jbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
add rax, rdx 
add rdx, rdi 
xor rcx, rdx 
xor rbx, rbx 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
cmp rbx, rcx 
mov rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
mov rdx, rsi 
cmp rcx, rbx 
mov rdx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
