.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rbx, qword ptr [rbx + rbx + 1] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4848 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
xor rdx, rdi 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
mov rsi, rax 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rbx] 
lea rdx, qword ptr [rcx + rdx + 1] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2160 
mov rdx, rax 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 8 
xor rbx, rdi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5648 
mov rsi, 6024 
xor rdx, rdi 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rax 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi 
mov rcx, 2568 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rsi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx 
lea rdi, qword ptr [rcx + rdi + 1] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5776 
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi] 
lea rdx, qword ptr [rdi + rdx + 1] 
lea rbx, qword ptr [rcx + rbx + 1] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7824 
mov rdx, rdi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi 
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 7904 
lea rdx, qword ptr [rdi + rdx + 1] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
