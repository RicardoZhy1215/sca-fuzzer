.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rax, rdi 
xor rbx, rax 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
xor rbx, rcx 
mov rsi, rbx 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi 
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx] 
lea rdx, qword ptr [rax + rdx + 1] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
lea rdi, qword ptr [rsi + rdi + 1] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi 
mov rsi, 6528 
mov rsi, rcx 
xor rdx, rsi 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3472 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
mov rdx, 6624 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
mov rdx, rsi 
mov rsi, rbx 
xor rsi, rsi 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
xor rbx, rsi 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 184 
xor rcx, rsi 
mov rcx, rbx 
xor rdx, rcx 
xor rcx, rsi 
lea rsi, qword ptr [rcx + rsi + 1] 
mov rcx, rsi 
lea rbx, qword ptr [rbx + rbx + 1] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
