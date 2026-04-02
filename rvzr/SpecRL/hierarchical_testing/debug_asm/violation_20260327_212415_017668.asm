.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -116 # instrumentation
mov rax, rdi 
mov rcx, rdx 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3 
xor rdx, rdi 
cmp rsi, rdx 
xor rdi, rdi 
mov rdx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
