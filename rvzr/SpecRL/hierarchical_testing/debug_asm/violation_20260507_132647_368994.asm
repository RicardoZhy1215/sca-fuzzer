.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx] 
xor rsi, rcx 
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5088 
lea rdi, qword ptr [rdi + rdi + 1] 
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi 
mov rbx, rdx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 7480 
mov rdx, 7176 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4648 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
