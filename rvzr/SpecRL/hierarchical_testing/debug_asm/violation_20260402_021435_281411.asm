.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -105 # instrumentation
cmp rsi, rbx 
cmp rbx, rax 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
add rdx, rcx 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 7 
add rcx, rdi 
add rdi, rdx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rax 
jnbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
add rcx, rsi 
mov rcx, rbx 
cmp rbx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
