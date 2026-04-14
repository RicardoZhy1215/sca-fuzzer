.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 65 # instrumentation
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 5 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 5 
mov rax, 1 
add rax, rdi 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
cmp rdx, rdi 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
jp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
mov rdi, rbx 
add rax, rdi 
and rsi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], 2 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rcx 
add rax, rsi 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
