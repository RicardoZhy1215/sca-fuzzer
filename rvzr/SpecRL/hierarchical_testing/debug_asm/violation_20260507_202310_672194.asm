.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdx 
lea rax, qword ptr [rdx + rax + 1] 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rax 
xor rax, rax 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5232 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
mov rdx, 144 
mov rsi, rdi 
lea rax, qword ptr [rsi + rax + 1] 
mov rax, rsi 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
