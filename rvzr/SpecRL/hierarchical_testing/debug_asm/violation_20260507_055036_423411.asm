.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rsi, rdi 
lea rsi, qword ptr [rdi + rsi + 1] 
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx] 
lea rsi, qword ptr [rsi + rsi + 1] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rcx 
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rax 
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7688 
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx] 
mov rbx, 704 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
