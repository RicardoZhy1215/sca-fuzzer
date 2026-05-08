.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rcx 
mov rdx, rbx 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
xor rdx, rsi 
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax] 
mov rcx, rsi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 200 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi 
mov rsi, rcx 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx 
lea rax, qword ptr [rcx + rax + 1] 
xor rdx, rdx 
mov rax, 3904 
mov rdx, rax 
xor rsi, rax 
lea rcx, qword ptr [rcx + rcx + 1] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdi 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx 
lea rcx, qword ptr [rsi + rcx + 1] 
lea rdi, qword ptr [rcx + rdi + 1] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3392 
xor rdx, rsi 
mov rdx, 2800 
lea rbx, qword ptr [rcx + rbx + 1] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5944 
and rdi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdi] 
lea rax, qword ptr [rax + rax + 1] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax 
mov rdi, 6208 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 7656 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
lea rax, qword ptr [rsi + rax + 1] 
xor rdx, rdi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4744 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
xor rdx, rdi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 3768 
lea rdx, qword ptr [rdi + rdx + 1] 
xor rbx, rcx 
mov rbx, rdx 
xor rbx, rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
