.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx] 
mov rdi, rsi 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdx 
mov rax, rbx 
add rdi, 1 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 1 
add rbx, 5 
cmp rbx, rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5 
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
mov rcx, 1 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rcx 
xor rsi, rdi 
cmp rcx, rdx 
sbb rax, 7 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
xor rcx, rax 
sbb rdi, rax 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rsi 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], 5 
mov rdi, rbx 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdi 
mov rdi, rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
