.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -63 # instrumentation
mov rcx, rbx 
cmp rax, rbx 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rcx 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rbx 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1 
mov rdi, rsi 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdx 
cmp rdi, rsi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4 
mov rbx, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
