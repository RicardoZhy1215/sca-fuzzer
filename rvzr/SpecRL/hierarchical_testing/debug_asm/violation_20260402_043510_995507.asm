.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -4 # instrumentation
add rdx, 6 
cmp rbx, rdi 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rax, rbx 
cmp rcx, rbx 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
mov rsi, rdx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
