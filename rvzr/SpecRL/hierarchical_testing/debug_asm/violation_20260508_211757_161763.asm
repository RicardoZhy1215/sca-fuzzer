.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rcx, qword ptr [rdi + rcx + 1] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax 
mov rax, 4936 
and rsi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rsi] 
mov rdi, 6536 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5192 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
