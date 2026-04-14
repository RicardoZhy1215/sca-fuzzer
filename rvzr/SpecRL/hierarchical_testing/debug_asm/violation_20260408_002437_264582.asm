.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 82 # instrumentation
mov rdi, rcx 
sbb rax, 5 
mov rdx, rcx 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 5 
add rax, 1 
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rax 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rax, rsi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rbx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rbx 
sbb rax, rbx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 7 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
