.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1520 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx] 
lea rbx, qword ptr [rdi + rbx + 1] 
mov rax, rsi 
lea rsi, qword ptr [rcx + rsi + 1] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 40 
lea rsi, qword ptr [rsi + rsi + 1] 
lea rax, qword ptr [rbx + rax + 1] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rcx 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
xor rdi, rbx 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi 
lea rdi, qword ptr [rcx + rdi + 1] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rbx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi 
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi] 
xor rdx, rax 
mov rdi, 5464 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rbx 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi 
xor rdi, rsi 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
xor rbx, rbx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx 
and rsi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rsi] 
mov rcx, rsi 
mov rcx, 96 
xor rcx, rdx 
mov rcx, rdi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx 
mov rdx, 1808 
xor rsi, rdi 
xor rcx, rbx 
xor rcx, rbx 
mov rcx, rsi 
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 7832 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
xor rcx, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
