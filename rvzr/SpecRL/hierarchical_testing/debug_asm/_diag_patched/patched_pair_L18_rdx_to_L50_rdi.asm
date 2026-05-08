.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi] 
lea rbx, qword ptr [rbx + rbx + 1] 
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + 8192], rdx  # CHECKPOINT save store base (rdx)
mov qword ptr [r14 + rdx], rcx 
mov rcx, 4416 
mov rdx, rax 
and rsi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rsi] 
lea rsi, qword ptr [rbx + rsi + 1] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 6456 
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi] 
mov rcx, rax 
xor rcx, rax 
mov rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + 8200], rdi  # CHECKPOINT save load base (rdi)
mov rcx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6528 
lea rbx, qword ptr [rsi + rbx + 1] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rax 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdx 
lea rdi, qword ptr [rdi + rdi + 1] 
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rbx 
xor rax, rdi 
lea rax, qword ptr [rax + rax + 1] 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6112 
mov rcx, 5592 
.exit_0:
mov rax, qword ptr [r14 + 8192]  # CHECKPOINT recover store base into rax
mov rbx, qword ptr [r14 + 8200]  # CHECKPOINT recover load base into rbx
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
