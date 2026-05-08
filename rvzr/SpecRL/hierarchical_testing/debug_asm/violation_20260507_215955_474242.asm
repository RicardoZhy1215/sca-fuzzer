.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdx 
mov rbx, 4856 
mov rsi, 7728 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi 
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2536 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
