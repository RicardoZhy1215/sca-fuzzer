.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -59 # instrumentation
mov rbx, rcx 
cmp rdi, rcx 
xor rax, rdx 
cmp rsi, rdi 
mov rcx, rax 
mov rcx, rsi 
add rdi, rax 
add rax, rbx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rax, rbx 
cmp rbx, rdi 
xor rax, rax 
add rcx, rdx 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
