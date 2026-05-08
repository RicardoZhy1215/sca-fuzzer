.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 6520 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7168 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx 
and rcx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rcx] 
xor rdi, rsi 
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx 
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx] 
xor rdi, rcx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4416 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx] 
lea rax, qword ptr [rcx + rax + 1] 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
xor rdi, rsi 
xor rcx, rdx 
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
lea rdx, qword ptr [rdx + rdx + 1] 
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7568 
xor rsi, rax 
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
