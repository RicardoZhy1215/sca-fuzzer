.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -31 # instrumentation
add rax, rdi 
add rdi, rdx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rsi 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdi 
jns .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rsi, rbx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
cmp rsi, rbx 
mov rbx, rdi 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rdx 
cmp rsi, rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
