.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rax, 5824 
mov rsi, rdi 
mov rcx, rax 
xor rax, rsi 
lea rax, qword ptr [rsi + rax + 1] 
lea rsi, qword ptr [rdi + rsi + 1] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
mov rcx, rsi 
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdi 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
mov rax, 3208 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax 
mov rax, rbx 
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
mov rcx, rbx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rbx 
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
