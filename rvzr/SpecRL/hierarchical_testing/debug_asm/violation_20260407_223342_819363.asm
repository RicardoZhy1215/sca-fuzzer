.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 81 # instrumentation
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx] 
add rdi, 6 
add rdx, rbx 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rax 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
mov rax, rdi 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
mov rdx, 3 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
and rsi, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdi 
sbb rsi, 0 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
mov rdx, 1 
add rax, rcx 
and rdi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdi] 
add rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
