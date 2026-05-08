.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -28 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 168 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdx 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
xor rax, rbx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdx 
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi] 
mov rcx, rdx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2872 
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
