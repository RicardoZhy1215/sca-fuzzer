.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rdx, qword ptr [rsi + rdx + 1] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx 
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
mov rbx, 2984 
and rbx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
mov rsi, 7560 
and rcx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi 
xor rcx, rsi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4656 
and rdi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdi] 
xor rdx, rsi 
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
