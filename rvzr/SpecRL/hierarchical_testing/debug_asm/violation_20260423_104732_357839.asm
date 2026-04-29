.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rsi, rdi 
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax] 
mov rbx, 6672 
mov rax, rdx 
mov rax, 3200 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rcx 
mov rbx, rax 
xor rsi, rcx 
xor rdx, rax 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rcx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdx 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
mov rdx, rax 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
xor rax, rax 
mov rdi, 6400 
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax] 
lea rcx, qword ptr [rdx + rcx + 1] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rcx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax 
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
mov rcx, 5328 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rax 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
mov rdi, 4024 
mov rdi, 1848 
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
mov rdx, rbx 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
