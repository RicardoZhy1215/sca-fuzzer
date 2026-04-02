.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -58 # instrumentation
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
jbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rsi 
add rdi, rax 
xor rdi, rdx 
cmp rcx, rax 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
