.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
mov rdx, rdi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 3208 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rbx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3632 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 760 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax 
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi 
xor rdi, rdi 
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi] 
mov rcx, rdi 
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
