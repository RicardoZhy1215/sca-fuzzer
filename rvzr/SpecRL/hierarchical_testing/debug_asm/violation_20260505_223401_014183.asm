.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 65 # instrumentation
mov rax, 6216 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdx 
mov rdi, rbx 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
jns .bb_0.1 
jmp .exit_0 
.bb_0.1:
xor rsi, rax 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
mov rax, rdx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi 
and rdx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7696 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
