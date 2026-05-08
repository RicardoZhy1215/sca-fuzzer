.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rcx, 1392 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi 
xor rax, rsi 
and rdi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdi] 
xor rbx, rsi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 536 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
