.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 976 
lea rsi, qword ptr [rdi + rsi + 1] 
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx] 
xor rdi, rcx 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
