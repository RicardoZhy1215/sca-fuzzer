.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi 
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
lea rdi, qword ptr [rax + rdi + 1] 
xor rdx, rcx 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
