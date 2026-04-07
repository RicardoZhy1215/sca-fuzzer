.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 49 # instrumentation
mov rax, rsi 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rax 
jp .bb_0.1 
jmp .exit_0 
.bb_0.1:
add rax, rsi 
add rcx, rdi 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
mov rbx, rsi 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
