.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rcx, qword ptr [rdi + rcx + 1] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax 
mov rax, 4936 
and rsi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rsi] 
mov rdi, 6536 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5192 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3472 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi 
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4928 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3480 
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4520 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1416 
lea rsi, qword ptr [rdi + rsi + 1] 
xor rdx, rcx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7696 
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx] 
xor rdx, rbx 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
lea rcx, qword ptr [rcx + rcx + 1] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rax 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
lea rcx, qword ptr [rsi + rcx + 1] 
xor rdx, rdi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rsi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3928 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5816 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6096 
xor rsi, rsi 
lea rbx, qword ptr [rcx + rbx + 1] 
mov rcx, 7896 
lea rdx, qword ptr [rdx + rdx + 1] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rcx 
and rcx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rcx] 
xor rbx, rbx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 2560 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 560 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
