.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx 
lea rcx, qword ptr [rax + rcx + 1] 
mov rax, 4056 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
lea rcx, qword ptr [rdx + rcx + 1] 
mov rsi, rbx 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi 
mov rdx, rcx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
mov rbx, rdx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7400 
mov rbx, rsi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 6904 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rcx 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
lea rax, qword ptr [rax + rax + 1] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdi 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rbx 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
mov rdi, rbx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 136 
xor rcx, rcx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3 
lea rax, qword ptr [rsi + rax + 1] 
lea rbx, qword ptr [rdx + rbx + 1] 
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
lea rcx, qword ptr [rbx + rcx + 1] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
xor rcx, rax 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
lea rcx, qword ptr [rdx + rcx + 1] 
lea rcx, qword ptr [rcx + rcx + 1] 
xor rax, rdi 
xor rdx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
