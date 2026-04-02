.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -43 # instrumentation
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
mov rcx, rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rsi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
xor rdi, rdx 
mov rsi, rbx 
xor rdx, rax 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
cmp rcx, rbx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdx 
mov rax, rbx 
mov rdx, rax 
xor rsi, rdi 
cmp rcx, rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rdi, rsi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rcx, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
xor rsi, rdi 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rsi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rsi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rax, rdx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
