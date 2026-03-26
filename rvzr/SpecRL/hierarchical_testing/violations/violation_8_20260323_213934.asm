.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 36 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rbx, rsi 
mov rdx, rax 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rbx 
xor rbx, rdx 
add rax, rbx 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
cmp rdx, rdi 
xor rax, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
