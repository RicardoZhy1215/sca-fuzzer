.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 392 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
mov rsi, rbx 
mov rdx, 4888 
and rcx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rcx] 
mov rdx, rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
