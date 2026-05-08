.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rbx, qword ptr [rax + rbx + 1] 
lea rsi, qword ptr [rax + rsi + 1] 
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 6392 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rax 
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdi 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3224 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
