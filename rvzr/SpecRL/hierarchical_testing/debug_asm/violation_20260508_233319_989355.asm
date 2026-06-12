.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5200 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rsi 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 752 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx 
mov rbx, rsi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rcx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2544 
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx] 
mov rsi, 208 
lea rax, qword ptr [rdx + rax + 1] 
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx] 
mov rbx, rdx 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx] 
xor rax, rbx 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
mov rdi, 4 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
