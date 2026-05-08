.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rcx, qword ptr [rsi + rcx + 1] 
mov rdi, 3552 
mov rcx, 360 
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi] 
mov rax, rbx 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
mov rax, 5336 
lea rax, qword ptr [rdx + rax + 1] 
and rdi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
