.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4432 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdi 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi 
xor rbx, rax 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
mov rsi, 6048 
mov rsi, 848 
xor rdx, rdx 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rbx 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
mov rcx, 6352 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx 
lea rdi, qword ptr [rsi + rdi + 1] 
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + 8192], rcx  # CHECKPOINT save store base (rcx)
mov qword ptr [r14 + rcx], rax 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdx 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
mov rbx, rax 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
mov rcx, 2896 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + 8200], rdx  # CHECKPOINT save load base (rdx)
mov rbx, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdx 
and rsi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rsi] 
xor rbx, rbx 
mov rbx, 6600 
xor rdx, rax 
and rcx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rcx] 
mov rax, rcx 
mov rdx, 2640 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx] 
mov rdi, rbx 
mov rdx, rsi 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rcx 
.exit_0:
mov rax, qword ptr [r14 + 8192]  # CHECKPOINT recover store base into rax
mov rbx, qword ptr [r14 + 8200]  # CHECKPOINT recover load base into rbx
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
