.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rax, qword ptr [rcx + rax + 1] 
lea rsi, qword ptr [rax + rsi + 1] 
mov rcx, 7792 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
mov rcx, rdx 
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi] 
mov rcx, 3032 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
mov rbx, rax 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdx 
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi 
mov rdi, 7808 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
mov rcx, rdx 
mov rdi, 1520 
lea rdx, qword ptr [rbx + rdx + 1] 
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdx 
mov rbx, rcx 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4152 
lea rcx, qword ptr [rbx + rcx + 1] 
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7552 
lea rdx, qword ptr [rbx + rdx + 1] 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rcx] 
xor rbx, rax 
and rdx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdx] 
mov rsi, rdi 
lea rsi, qword ptr [rbx + rsi + 1] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
