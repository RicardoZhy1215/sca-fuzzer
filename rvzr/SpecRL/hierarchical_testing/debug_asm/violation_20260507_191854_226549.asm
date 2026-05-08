.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 976 
lea rsi, qword ptr [rdi + rsi + 1] 
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx] 
xor rdi, rcx 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdx 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdx 
lea rdx, qword ptr [rax + rdx + 1] 
lea rcx, qword ptr [rbx + rcx + 1] 
and rsi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rsi] 
xor rdi, rdx 
and rbx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rsi 
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 968 
lea rsi, qword ptr [rdi + rsi + 1] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdx 
mov rax, 4608 
lea rbx, qword ptr [rax + rbx + 1] 
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 304 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi 
mov rdx, 3984 
mov rsi, rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
