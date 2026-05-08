.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
xor rbx, rax 
and rdx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx 
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
xor rbx, rdx 
lea rsi, qword ptr [rax + rsi + 1] 
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7880 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5768 
and rdx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdx] 
xor rbx, rbx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 6 
xor rsi, rcx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5424 
mov rbx, rdx 
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi] 
lea rax, qword ptr [rcx + rax + 1] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi 
lea rdi, qword ptr [rdi + rdi + 1] 
mov rbx, 5080 
mov rcx, 1624 
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4720 
mov rdi, rsi 
lea rsi, qword ptr [rdi + rsi + 1] 
lea rsi, qword ptr [rax + rsi + 1] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax 
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
mov rcx, rdx 
lea rbx, qword ptr [rdx + rbx + 1] 
mov rdi, 6072 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
