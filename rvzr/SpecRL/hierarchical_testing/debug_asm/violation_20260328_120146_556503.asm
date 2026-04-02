.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -113 # instrumentation
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rcx 
xor rdx, rbx 
mov rsi, rbx 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx 
mov rsi, rdx 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
mov rax, rbx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
cmp rdx, rax 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
jl .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rbx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
cmp rdi, rax 
mov rax, rdi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
