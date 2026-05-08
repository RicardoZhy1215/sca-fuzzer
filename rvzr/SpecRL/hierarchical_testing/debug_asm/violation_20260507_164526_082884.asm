.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rdi, rsi 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
lea rsi, qword ptr [rdx + rsi + 1] 
lea rcx, qword ptr [rsi + rcx + 1] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4168 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdx 
xor rdi, rdi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4704 
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
