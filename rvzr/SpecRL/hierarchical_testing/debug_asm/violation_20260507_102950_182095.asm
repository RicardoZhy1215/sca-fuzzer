.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rax, qword ptr [rcx + rax + 1] 
lea rsi, qword ptr [rax + rsi + 1] 
mov rcx, 7792 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
mov rcx, rdx 
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi] 
mov rcx, 3032 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
mov rbx, rax 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdx 
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
