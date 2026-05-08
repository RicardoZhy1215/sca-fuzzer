.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 392 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
mov rsi, rbx 
mov rdx, 4888 
and rcx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rcx] 
mov rdx, rcx 
mov rdi, 3488 
mov rcx, rax 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rcx 
lea rax, qword ptr [rdx + rax + 1] 
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
mov rax, rbx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6296 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7600 
lea rbx, qword ptr [rbx + rbx + 1] 
xor rbx, rax 
xor rdx, rdx 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi 
lea rdi, qword ptr [rbx + rdi + 1] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3600 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rax 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2496 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
xor rbx, rax 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
and rdx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4160 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
