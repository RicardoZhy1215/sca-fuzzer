.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rdx, rsi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx 
xor rbx, rdi 
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rcx 
mov rdx, rax 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
mov rsi, rdi 
xor rdx, rdx 
lea rcx, qword ptr [rsi + rcx + 1] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax 
and rsi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
lea rcx, qword ptr [rsi + rcx + 1] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 328 
lea rbx, qword ptr [rbx + rbx + 1] 
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
mov rax, rdx 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx] 
lea rdi, qword ptr [rdx + rdi + 1] 
xor rdx, rsi 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rcx 
mov rdi, 5104 
mov rsi, rdi 
and rbx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rbx] 
xor rcx, rdx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6528 
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx] 
mov rbx, rcx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4056 
xor rdx, rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
