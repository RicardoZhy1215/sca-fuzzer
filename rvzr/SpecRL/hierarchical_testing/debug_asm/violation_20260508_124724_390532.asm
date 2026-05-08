.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rax, rsi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5144 
xor rax, rsi 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
lea rdx, qword ptr [rdx + rdx + 1] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1272 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
mov rax, 3072 
xor rdx, rcx 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
mov rdi, rbx 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 808 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx 
lea rax, qword ptr [rsi + rax + 1] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
