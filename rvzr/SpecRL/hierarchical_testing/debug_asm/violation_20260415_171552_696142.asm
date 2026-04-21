.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -70 # instrumentation
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], 0 
mov rdx, 6 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rax 
add rbx, 2 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdx 
add rdi, rax 
sbb rdx, 5 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdi 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rcx 
jl .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
sbb rsi, 0 
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax] 
cmp rdx, rdi 
add rdi, rbx 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], 0 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
