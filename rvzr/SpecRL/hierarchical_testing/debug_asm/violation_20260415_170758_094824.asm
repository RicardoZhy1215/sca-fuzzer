.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 10 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdx 
mov rdx, 2 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
and rdx, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 0 
jle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 0 
add rbx, 4 
add rsi, 1 
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rbx 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 3 
and rsi, 0b1111111111111 # instrumentation
sbb rsi, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rax 
mov rdi, 5 
and rbx, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rbx] 
mov rax, 2 
mov rsi, rax 
mov rdx, rsi 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6 
xor rbx, rcx 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rax 
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3 
add rdx, 1 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
