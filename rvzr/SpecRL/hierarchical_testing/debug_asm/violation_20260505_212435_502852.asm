.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 11 # instrumentation
mov rcx, 4968 
mov rbx, 8152 
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rsi, rax 
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
