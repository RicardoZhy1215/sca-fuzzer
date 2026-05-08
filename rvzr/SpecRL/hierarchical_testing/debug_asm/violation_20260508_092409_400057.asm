.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
lea rcx, qword ptr [rax + rcx + 1] 
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4472 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5560 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 72 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
xor rbx, rcx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2384 
mov rsi, rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
