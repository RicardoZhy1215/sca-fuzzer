.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi 
mov rsi, 2496 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3096 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2120 
mov rcx, rax 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6472 
lea rsi, qword ptr [rax + rsi + 1] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 6936 
lea rbx, qword ptr [rsi + rbx + 1] 
mov rsi, 7408 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
